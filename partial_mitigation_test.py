from qcg.generators import gen_supremacy, gen_hwea
from helper_fun import get_evaluator_info, cross_entropy, reverseBits, apply_measurement
from qiskit import Aer, IBMQ, execute
from time import time
import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile, assemble
import matplotlib.pyplot as plt

def apply_mitigation(raw,meas_fitter):
    num_mitigated_qubits = int(np.log2(len(meas_fitter.cal_matrix)))
    print(num_mitigated_qubits)

def evaluate_circ(circ, backend, evaluator_info):
    if backend == 'statevector_simulator':
        # print('using statevector simulator')
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_ordered = [0 for sv in outputstate]
        for i, sv in enumerate(outputstate):
            reverse_i = reverseBits(i,len(circ.qubits))
            outputstate_ordered[reverse_i] = sv
        sv_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return sv_prob
    elif backend == 'noiseless_qasm_simulator':
        # print('using noiseless qasm simulator %d shots'%num_shots)
        backend = Aer.get_backend('qasm_simulator')
        qc = apply_measurement(circ)

        num_shots = evaluator_info['num_shots']
        noiseless_qasm_result = execute(qc, backend, shots=num_shots).result()
        noiseless_counts = noiseless_qasm_result.get_counts(qc)
        noiseless_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noiseless_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
        return noiseless_prob
    elif backend == 'noisy_qasm_simulator':
        # print('using noisy qasm simulator {} shots'.format(num_shots))
        backend = Aer.get_backend('qasm_simulator')
        qc=apply_measurement(circ)
        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'], 
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])
        noisy_qasm_result = execute(experiments=mapped_circuit,
        backend=backend,
        noise_model=evaluator_info['noise_model'],
        coupling_map=evaluator_info['coupling_map'],
        basis_gates=evaluator_info['basis_gates'],
        shots=evaluator_info['num_shots']).result()
        if 'meas_filter' in evaluator_info:
            noisy_qasm_result = evaluator_info['meas_filter'].apply(noisy_qasm_result)
        # elif 'meas_fitter' in evaluator_info:
        #     noisy_qasm_result = apply_mitigation(noisy_qasm_result,meas_fitter)
        noisy_counts = noisy_qasm_result.get_counts(qc)
        print(noisy_counts)
        noisy_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noisy_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noisy_prob[reversed_state] = noisy_counts[state]/evaluator_info['num_shots']
        return noisy_prob
    elif backend == 'hardware':
        qc=apply_measurement(circ)

        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])

        device_max_shots = evaluator_info['device'].configuration().max_shots
        remaining_shots = evaluator_info['num_shots']
        hw_counts = {}
        while remaining_shots>0:
            batch_shots = min(remaining_shots,device_max_shots)
            qobj = assemble(mapped_circuit, backend=evaluator_info['device'], shots=batch_shots)
            print('Submitted %d shots to hardware'%(batch_shots))
            job = evaluator_info['device'].run(qobj)
            hw_result = job.result()
            if 'meas_filter' in evaluator_info:
                print('Mitigation for %d qubit circuit'%(len(circ.qubits)))
                mitigation_begin = time()
                mitigated_results = evaluator_info['meas_filter'].apply(hw_result)
                hw_counts_batch = mitigated_results.get_counts(0)
                print('Mitigation for %d qubit circuit took %.3e seconds'%(len(circ.qubits),time()-mitigation_begin))
            else:
                hw_counts_batch = hw_result.get_counts(qc)
            for state in hw_counts_batch:
                if state not in hw_counts:
                    hw_counts[state] = hw_counts_batch[state]
                else:
                    hw_counts[state] += hw_counts_batch[state]
            remaining_shots -= batch_shots
        
        hw_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in hw_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            hw_prob[reversed_state] = hw_counts[state]/evaluator_info['num_shots']
        return hw_prob
    else:
        raise Exception('Illegal backend :',backend)

def partial_readout_mitigation(evaluator_info):
    filter_begin = time()
    device = evaluator_info['device']
    num_shots = device.configuration().max_shots
    properties = evaluator_info['properties']
    num_qubits = len(properties.qubits)
    max_group_len = int(np.log2(device.configuration().max_experiments/2))
    max_group_len = 1

    # Generate the calibration circuits
    qr = QuantumRegister(num_qubits)
    qubit_list = []
    _initial_layout = evaluator_info['initial_layout'].get_physical_bits()
    for q in _initial_layout:
        if 'ancilla' not in _initial_layout[q].register.name and len(qubit_list)<max_group_len:
            qubit_list.append(q)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')
    print(evaluator_info['initial_layout'])
    print('Calibrating for:',qubit_list)
    print('Calculating measurement filter, %d-qubit calibration circuits * %d * %.3e shots.'%(len(meas_calibs[0].qubits),len(meas_calibs),num_shots),end=' ')
    assert len(meas_calibs)<=device.configuration().max_experiments/2

    # Execute the calibration circuits
    meas_calibs_transpiled = transpile(meas_calibs, backend=device)
    cal_results = execute(experiments=meas_calibs_transpiled,
        backend=Aer.get_backend('qasm_simulator'),
        coupling_map=evaluator_info['coupling_map'],
        noise_model=evaluator_info['noise_model'],
        basis_gates=evaluator_info['basis_gates'],
        shots=evaluator_info['num_shots']).result()

    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
    meas_filter = meas_fitter.filter
    filter_time = time() - filter_begin
    print('%.3e seconds'%filter_time)
    return meas_filter, meas_fitter

circ = gen_hwea(4,1)

ground_truth = evaluate_circ(circ=circ, backend='statevector_simulator', evaluator_info=None)
print('ground truth:',ground_truth)

evaluator_info = get_evaluator_info(circ=circ,device_name='ibmq_boeblingen',fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model','num_shots'])
qasm_noise = evaluate_circ(circ=circ, backend='noisy_qasm_simulator', evaluator_info=evaluator_info)
print('No mitigation:', qasm_noise)

meas_filter, meas_fitter = partial_readout_mitigation(evaluator_info)
# evaluator_info['meas_filter'] = meas_filter
qasm_noise_mitigated = evaluate_circ(circ=circ, backend='noisy_qasm_simulator', evaluator_info=evaluator_info)
print('With mitigation:', qasm_noise_mitigated)

print(meas_fitter.cal_matrix)
meas_fitter.plot_calibration()
print(len(meas_fitter.cal_matrix))

plot_range = min(64,len(ground_truth))
x = np.arange(len(ground_truth))[:plot_range]

plt.figure()
plt.bar(x,height=ground_truth[:plot_range],label='ground truth, %.3e'%cross_entropy(ground_truth,ground_truth))
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.savefig('./plots/ground_truth.png',dpi=400)

plt.figure()
plt.bar(x,height=qasm_noise_mitigated[:plot_range],label='mitigated, %.3e'%cross_entropy(ground_truth,qasm_noise_mitigated))
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.savefig('./plots/mitigated.png',dpi=400)

plt.figure()
plt.bar(x,height=qasm_noise[:plot_range],label='no mitigation, %.3e'%cross_entropy(ground_truth,qasm_noise))
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.savefig('./plots/vanilla.png',dpi=400)