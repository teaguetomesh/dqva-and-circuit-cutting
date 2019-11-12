from qcg.generators import gen_supremacy, gen_hwea
from helper_fun import evaluate_circ, get_evaluator_info, cross_entropy
from qiskit import Aer, IBMQ, execute
from time import time
import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile, assemble
import matplotlib.pyplot as plt

def readout_mitigation(evaluator_info):
    num_shots = evaluator_info['num_shots']
    device = evaluator_info['device']
    initial_layout = evaluator_info['initial_layout']
    if num_shots>device.configuration().max_shots:
        print('During readout mitigation, num_shots %.3e exceeded hardware max'%num_shots)
        num_shots = device.configuration().max_shots
    filter_begin = time()
    properties = device.properties()
    num_qubits = len(properties.qubits)

    # Generate the calibration circuits
    qr = QuantumRegister(num_qubits)
    qubit_list = []
    _initial_layout = initial_layout.get_physical_bits()
    for q in _initial_layout:
        if 'ancilla' not in _initial_layout[q].register.name:
            qubit_list.append(q)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')
    print('Calculating measurement filter, %d-qubit calibration circuits * %d * %.3e shots.'%(len(meas_calibs[0].qubits),len(meas_calibs),num_shots),end=' ')
    assert len(meas_calibs)<=device.configuration().max_experiments/3*2

    # Execute the calibration circuits
    meas_calibs_transpiled = transpile(meas_calibs, backend=device)
    device = Aer.get_backend('qasm_simulator')
    cal_results = execute(experiments=meas_calibs_transpiled,
        backend=device,
        coupling_map=evaluator_info['coupling_map'],
        noise_model=evaluator_info['noise_model'],
        basis_gates=evaluator_info['basis_gates'],
        shots=evaluator_info['num_shots']).result()

    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
    meas_filter = meas_fitter.filter
    filter_time = time() - filter_begin
    print('%.3e seconds'%filter_time)
    return meas_filter, meas_fitter

def partial_readout_mitigation(evaluator_info):
    device = evaluator_info['device']
    num_shots = evaluator_info['num_shots']
    if num_shots>device.configuration().max_shots:
        print('During readout mitigation, num_shots %.3e exceeded hardware max'%num_shots)
        num_shots = device.configuration().max_shots
    filter_begin = time()
    properties = evaluator_info['properties']
    num_qubits = len(properties.qubits)
    max_group_len = int(np.log2(device.configuration().max_experiments/2))
    max_group_len = 3

    # Generate the calibration circuits
    qr = QuantumRegister(num_qubits)
    qubit_list = []
    _initial_layout = evaluator_info['initial_layout'].get_physical_bits()
    for q in _initial_layout:
        if 'ancilla' not in _initial_layout[q].register.name and len(qubit_list)<max_group_len:
            qubit_list.append(q)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')
    print('Calculating measurement filter, %d-qubit calibration circuits * %d * %.3e shots.'%(len(meas_calibs[0].qubits),len(meas_calibs),num_shots),end=' ')
    assert len(meas_calibs)<=device.configuration().max_experiments/2

    # Execute the calibration circuits
    device = Aer.get_backend('qasm_simulator')
    meas_calibs_transpiled = transpile(meas_calibs, backend=device)
    cal_results = execute(experiments=meas_calibs_transpiled,
        backend=device,
        coupling_map=evaluator_info['coupling_map'],
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

meas_filter, meas_fitter = readout_mitigation(evaluator_info)
evaluator_info['meas_filter'] = meas_filter
qasm_noise_mitigated = evaluate_circ(circ=circ, backend='noisy_qasm_simulator', evaluator_info=evaluator_info)
print('With mitigation:', qasm_noise_mitigated)

meas_fitter.plot_calibration()

plot_range = min(64,len(ground_truth))
x = np.arange(len(ground_truth))[:plot_range]

plt.figure()
plt.bar(x,height=ground_truth[:plot_range],label='ground truth, %.3e'%cross_entropy(ground_truth,ground_truth))
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.show()

plt.figure()
plt.bar(x,height=qasm_noise_mitigated[:plot_range],label='mitigated, %.3e'%cross_entropy(ground_truth,qasm_noise_mitigated))
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.show()

plt.figure()
plt.bar(x,height=qasm_noise[:plot_range],label='no mitigation, %.3e'%cross_entropy(ground_truth,qasm_noise))
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.show()