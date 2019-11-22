from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.compiler import transpile, assemble
from qiskit.providers.aer import noise
import numpy as np
import math
from datetime import datetime
from qiskit.providers.models import BackendProperties
from qiskit.ignis.mitigation.measurement import complete_meas_cal
import datetime as dt
import pickle
import copy
from time import time

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
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    sum_of_prob = sum(obs)
    obs = [x/sum_of_prob for x in obs]
    assert abs(sum(obs)-1) <= 1e-10
    h = 0
    for p,q in zip(target,obs):
        if p==0 or q<=0:
            h += 0
        else:
            h += -p*np.log(q)
    return h

def fidelity(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    sum_of_prob = sum(obs)
    obs = [x/sum_of_prob for x in obs]
    assert abs(sum(obs)-1) <= 1e-10
    fidelity = 0
    for t,o in zip(target,obs):
        if t!= 0:
            fidelity += o
    return fidelity

def find_cluster_O_rho_qubits(complete_path_map,cluster_idx):
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for q in path[:-1]:
                if q[0] == cluster_idx:
                    O_qubits.append(q)
            for q in path[1:]:
                if q[0] == cluster_idx:
                    rho_qubits.append(q)
    return O_qubits, rho_qubits

def get_circ_saturated_shots(circs,accuracy):
    saturated_shots = []
    for circ_idx, circ in enumerate(circs):
        ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)
        min_ce = cross_entropy(target=ground_truth,obs=ground_truth)
        qasm_prob = [0 for i in ground_truth]
        shots_increment = 1024
        evaluator_info = {}
        evaluator_info['num_shots'] = shots_increment
        counter = 0.0
        while 1:
            counter += 1.0
            qasm_prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=evaluator_info)
            qasm_prob = [(x*(counter-1)+y)/counter for x,y in zip(qasm_prob,qasm_prob_batch)]
            ce = cross_entropy(target=ground_truth,obs=qasm_prob)
            if min_ce != 0:
                diff = abs((ce-min_ce)/min_ce)
            else:
                diff = abs(ce-min_ce)
            if counter%50==49:
                print('current diff:',diff,'current shots:',int(counter*shots_increment))
            if diff < accuracy:
                saturated_shots.append(int(counter*shots_increment))
                # print('circ %d saturated shots = %d, ce difference = %.3f'%(circ_idx,saturated_shots[-1],diff))
                break
    return saturated_shots

def distribute_cluster_shots(total_shots,clusters,complete_path_map):
    cluster_shots = []
    for cluster_idx, cluster_circ in enumerate(clusters):
        O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
        num_instances = np.power(6,len(rho_qubits))*np.power(3,len(O_qubits))
        cluster_shots.append(math.ceil(total_shots/num_instances))
    return cluster_shots

def split_shots(total_shots,max_shots,max_experiments):
    if total_shots > max_shots*max_experiments:
        current_schedule = (max_experiments,max_shots)
        next_schedule = split_shots(total_shots=total_shots-max_shots*max_experiments,max_shots=max_shots,max_experiments=max_experiments)
        next_schedule.insert(0,current_schedule)
        return next_schedule
    elif total_shots == max_shots*max_experiments:
        return [(max_experiments,max_shots)]
    else:
        for num_experiments in range(1,max_experiments+1):
            equally_divided_shots = int(total_shots/num_experiments)
            if total_shots%num_experiments==0 and equally_divided_shots<=max_shots:
                return [(num_experiments,equally_divided_shots)]
            else:
                continue
        return None

def apply_measurement(circ):
    c = ClassicalRegister(len(circ.qubits), 'c')
    meas = QuantumCircuit(circ.qregs[0], c)
    meas.barrier(circ.qubits)
    meas.measure(circ.qubits,c)
    qc = circ+meas
    return qc

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

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
        assert 'meas_filter' not in evaluator_info
        noisy_counts = noisy_qasm_result.get_counts(qc)
        noisy_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noisy_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noisy_prob[reversed_state] = noisy_counts[state]/evaluator_info['num_shots']
        return noisy_prob
    elif backend == 'hardware':
        jobs = []
        qc=apply_measurement(circ)

        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])

        device_max_shots = evaluator_info['device'].configuration().max_shots
        device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/2)

        schedule = split_shots(total_shots=evaluator_info['num_shots'],max_shots=device_max_shots,max_experiments=device_max_experiments)
        
        for s in schedule:
            num_experiments, num_shots = s
            mapped_circuit_l = [mapped_circuit for i in range(num_experiments)]
            qobj = assemble(mapped_circuit_l, backend=evaluator_info['device'], shots=num_shots)
            print('Submitted %d * %d = %d shots to hardware'%(num_experiments,num_shots,num_experiments*num_shots))
            job = evaluator_info['device'].run(qobj)
            # job = execute(mapped_circuit_l, backend=Aer.get_backend('qasm_simulator'), shots=num_shots)
            jobs.append({'job':job,'circ':circ,'mapped_circuit_l':mapped_circuit_l,'evaluator_info':evaluator_info})
        return jobs
    else:
        raise Exception('Illegal backend :',backend)

# Entangled readout mitigation
def readout_mitigation(device,initial_layout):
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
    num_shots = device.configuration().max_shots
    # print('Calculating measurement filter, %d-qubit calibration circuits * %d * %.3e shots.'%(len(meas_calibs[0].qubits),len(meas_calibs),num_shots))
    assert len(meas_calibs)<=device.configuration().max_experiments/3*2

    # Execute the calibration circuits
    meas_calibs_transpiled = transpile(meas_calibs, backend=device)
    qobj = assemble(meas_calibs_transpiled, backend=device, shots=num_shots)
    job = device.run(qobj)
    # job = execute(meas_calibs_transpiled, backend=Aer.get_backend('qasm_simulator'), shots=num_shots)
    return job, state_labels, qubit_list

def get_evaluator_info(circ,device_name,fields):
    provider = load_IBMQ()
    device = provider.get_backend(device_name)
    properties = device.properties()
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties)
    basis_gates = noise_model.basis_gates
    _evaluator_info = {'device':device,
    'properties':properties,
    'coupling_map':coupling_map,
    'noise_model':noise_model,
    'basis_gates':basis_gates}
    
    if 'initial_layout' in fields:
        dag = circuit_to_dag(circ)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(dag)
        initial_layout = noise_mapper.property_set['layout']
        _evaluator_info['initial_layout'] = initial_layout

    evaluator_info = {}
    for field in fields:
        evaluator_info[field] = _evaluator_info[field]
    
    return evaluator_info