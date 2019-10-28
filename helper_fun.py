from qiskit import IBMQ
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.compiler import transpile
from qiskit.providers.aer import noise
import numpy as np
from datetime import datetime
from qiskit.providers.models import BackendProperties
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.circuit.quantumregister import QuantumRegister
import datetime as dt
import pickle

def load_IBMQ():
    token = '9056ff772ff2e0f19de847fc8980b6e0121b561832de7dfb72bb23b085c1dc4a62cde82392f7d74e655465a9d997dd970858a568434f1b97038e70bf44b6c8a6'
    if len(IBMQ.stored_account()) == 0:
        IBMQ.save_account(token)
        IBMQ.load_account()
    elif IBMQ.active_account() == None:
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-ornl', group='bes-qis', project='argonne')
    return provider

def cross_entropy(target,obs):
    assert len(target)==len(obs)
    # obs = [x if x>=0 else 0 for x in obs]
    alpha = 1e-14
    if 0 in obs:
        obs = [(x+alpha)/(1+alpha*len(obs)) for x in obs]
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            h += -p*np.log(q)
    return h

def find_saturated_shots(circ):
    ground_truth = simulate_circ(circ=circ,backend='statevector_simulator',qasm_info=None)
    min_ce = cross_entropy(target=ground_truth,obs=ground_truth)
    num_shots = 1000
    while 1:
        qasm = simulate_circ(circ=circ,backend='noiseless_qasm_simulator',qasm_info={'num_shots':num_shots})
        # NOTE: toggle here to control cross entropy accuracy
        if abs(cross_entropy(target=ground_truth,obs=qasm)-min_ce)/min_ce<1e-2:
            return num_shots
        else:
            num_shots += 1000

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def simulate_circ(circ, backend, qasm_info):
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
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas

        num_shots = qasm_info['num_shots']
        job_sim = execute(qc, backend, shots=num_shots)
        result = job_sim.result()
        noiseless_counts = result.get_counts(qc)
        noiseless_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noiseless_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
        return noiseless_prob
    elif backend == 'noisy_qasm_simulator':
        # print('using noisy qasm simulator {} shots'.format(num_shots))
        backend = Aer.get_backend('qasm_simulator')
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        device = qasm_info['device']
        properties = qasm_info['properties']
        coupling_map = qasm_info['coupling_map']
        noise_model = qasm_info['noise_model']
        basis_gates = qasm_info['basis_gates']
        num_shots = qasm_info['num_shots']
        meas_filter = qasm_info['meas_filter']
        dag = circuit_to_dag(qc)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(dag)
        initial_layout = noise_mapper.property_set['layout']
        new_circuit = transpile(qc, backend=device, basis_gates=basis_gates,coupling_map=coupling_map,backend_properties=properties,initial_layout=initial_layout)
        # bprob_noise_model = get_bprop()
        noisy_result = execute(experiments=new_circuit,
        backend=backend,
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        shots=num_shots).result()
        # mitigated_results = meas_filter.apply(noisy_result)
        # noisy_counts = mitigated_results.get_counts(0)
        noisy_counts = noisy_result.get_counts(new_circuit)
        noisy_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noisy_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noisy_prob[reversed_state] = noisy_counts[state]/num_shots
        return noisy_prob
    else:
        raise Exception('Illegal backend:',backend)

def get_bprop():
    public_provider = IBMQ.get_provider('ibm-q')
    mel = public_provider.get_backend('ibmq_16_melbourne')
    prop = mel.properties()
    qubit_list = prop.qubits[:]
    for i in range(len(qubit_list)):
        idx = -1
        for j,nduv in enumerate(qubit_list[i]):
            if nduv.name == 'readout_error':
                idx = j
                break
        if idx != -1:
            qubit_list[i][idx].value = 0.0
    calib_time = datetime(year=2019, month=10, day=15, hour=0, minute=0, second=0) #junk, set any time you like
    bprop = BackendProperties(last_update_date=calib_time, backend_name="no_readout_error", qubits=qubit_list, backend_version="1.0.0", gates=prop.gates, general=[])
    bprop_noise_model = noise.device.basic_device_noise_model(bprop)
    return bprop_noise_model

def readout_mitigation(circ,num_shots,device_name='ibmq_16_melbourne'):
    provider = load_IBMQ()
    device = provider.get_backend(device_name)
    properties = device.properties(dt.datetime(day=16, month=10, year=2019, hour=20))
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties)
    basis_gates = noise_model.basis_gates
    dag = circuit_to_dag(circ)
    noise_mapper = NoiseAdaptiveLayout(properties)
    noise_mapper.run(dag)
    initial_layout = noise_mapper.property_set['layout']
    num_qubits = len(properties.qubits)

    # Generate the calibration circuits
    qr = QuantumRegister(num_qubits)
    qubit_list = []
    # print(initial_layout)
    initial_layout = initial_layout.get_physical_bits()
    for q in initial_layout:
        if 'ancilla' not in initial_layout[q].register.name:
            qubit_list.append(q)
    # print(qubit_list)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')

    # Execute the calibration circuits without noise
    backend = Aer.get_backend('qasm_simulator')
    cal_results = execute(experiments=meas_calibs,
        backend=backend,
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        shots=num_shots).result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
    meas_filter = meas_fitter.filter
    return meas_filter