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
from qiskit.ignis.mitigation.measurement import complete_meas_cal
import pickle
import copy
from time import time
import os
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
from utils.conversions import list_to_dict, dict_to_array
from utils.metrics import chi2_distance
from scipy.stats import wasserstein_distance
import itertools

def generate_circ(full_circ_size,circuit_type):
    def gen_secret(num_qubit):
        num_digit = num_qubit-1
        num = 2**num_digit-1
        num = bin(num)[2:]
        num_with_zeros = str(num).zfill(num_digit)
        return num_with_zeros

    i,j = factor_int(full_circ_size)
    if circuit_type == 'supremacy':
        full_circ = gen_supremacy(i,j,8)
    elif circuit_type == 'hwea':
        full_circ = gen_hwea(i*j,1)
    elif circuit_type == 'bv':
        full_circ = gen_BV(gen_secret(i*j),barriers=False)
    elif circuit_type == 'qft':
        full_circ = gen_qft(width=i*j, barriers=False)
    elif circuit_type == 'sycamore':
        full_circ = gen_sycamore(i,j,8)
    else:
        raise Exception('Illegal circuit type:',circuit_type)
    return full_circ

def get_filename(experiment_name,circuit_type,device_name,field,evaluation_method=None):
    dirname = './{}/benchmark_data/{}_{}/'.format(experiment_name,circuit_type,device_name)
    if field == 'evaluator_input':
        evaluator_input_filename = 'evaluator_input_{}.p'.format(device_name)
        return dirname, evaluator_input_filename
    elif field == 'job_submittor_input':
        job_submittor_input_filename = 'job_submittor_input_{}.p'.format(device_name)
        return dirname, job_submittor_input_filename
    elif field == 'uniter_input':
        if evaluation_method == 'statevector_simulator':
            uniter_input_filename = 'classical_uniter_input_{}.p'.format(device_name)
        elif evaluation_method == 'noisy_qasm_simulator':
            uniter_input_filename = 'quantum_uniter_input_{}.p'.format(device_name)
        elif evaluation_method == 'hardware':
            uniter_input_filename = 'hw_uniter_input_{}.p'.format(device_name)
        elif evaluation_method == 'fake':
            uniter_input_filename = 'fake_uniter_input_{}.p'.format(device_name)
        else:
            raise Exception('Illegal evaluation method :',evaluation_method)
        return dirname, uniter_input_filename
    elif field == 'plotter_input':
        if evaluation_method == 'statevector_simulator':
            plotter_input_filename = 'classical_plotter_input_{}.p'.format(device_name)
        elif evaluation_method == 'noisy_qasm_simulator':
            plotter_input_filename = 'quantum_plotter_input_{}.p'.format(device_name)
        elif evaluation_method == 'hardware':
            plotter_input_filename = 'hw_plotter_input_{}.p'.format(device_name)
        elif evaluation_method == 'fake':
            plotter_input_filename = 'fake_plotter_input_{}.p'.format(device_name)
        else:
            raise Exception('Illegal evaluation method :',evaluation_method)
        return dirname, plotter_input_filename
    elif field == 'plotter_output':
        return dirname
    else:
        raise Exception('Illegal filename field :',field)

def read_file(filename):
    if os.path.isfile(filename):
        f = open(filename,'rb')
        file_content = {}
        while 1:
            try:
                file_content.update(pickle.load(f))
            except (EOFError):
                break
        f.close()
    else:
        file_content = {}
    return file_content

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n/val)
        if val*co_val == n:
            return val, co_val
        else:
            val -= 1

def load_IBMQ():
    token = '9056ff772ff2e0f19de847fc8980b6e0121b561832de7dfb72bb23b085c1dc4a62cde82392f7d74e655465a9d997dd970858a568434f1b97038e70bf44b6c8a6'
    if len(IBMQ.stored_account()) == 0:
        IBMQ.save_account(token)
        IBMQ.load_account()
    elif IBMQ.active_account() == None:
        # TODO: report warning triggered by load_account()
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-ornl', group='bes-qis', project='argonne')
    return provider

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

def get_circ_saturated_shots(circs,device_name):
    saturated_shots = []
    ground_truths = []
    saturated_probs = []
    for circ_idx, circ in enumerate(circs):
        full_circ_size = len(circ.qubits)
        if full_circ_size<10:
            min_saturated_shots = 8192
        elif full_circ_size<15:
            min_saturated_shots = 8192*10
        elif full_circ_size<18:
            min_saturated_shots = 8192*20
        else:
            min_saturated_shots = 8192*50
        ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None,force_prob=True)
        ground_truth = dict_to_array(distribution_dict=ground_truth,force_prob=True)

        qasm_evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,fields=['device'])
        device_max_shots = qasm_evaluator_info['device'].configuration().max_shots
        device_max_experiments = int(qasm_evaluator_info['device'].configuration().max_experiments/5)
        shots_increment = device_max_shots
        qasm_evaluator_info['num_shots'] = shots_increment

        chi2_l = []
        counter = 1
        accumulated_prob = np.zeros(2**full_circ_size,dtype=float)
        while 1:
            noiseless_prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info,force_prob=True)
            noiseless_prob_batch = dict_to_array(distribution_dict=noiseless_prob_batch,force_prob=True)
            accumulated_prob = ((counter-1)*accumulated_prob+noiseless_prob_batch)/counter
            assert abs(sum(accumulated_prob)-1)<1e-10
            accumulated_chi2 = chi2_distance(target=ground_truth,obs=accumulated_prob)
            # accumulated_chi2 = wasserstein_distance(u_values=ground_truth,v_values=accumulated_prob)
            # print('accumulated_chi2:',accumulated_chi2)
            chi2_l.append(accumulated_chi2)
            if len(chi2_l)>=3:
                accumulated_shots = int((len(chi2_l)-1)*shots_increment)
                first_derivative = (chi2_l[-1]+chi2_l[-3])/2
                second_derivative = (chi2_l[-1]+chi2_l[-3]-2*chi2_l[-2])/2
                if (abs(first_derivative)<1e-3 and abs(second_derivative)<1e-3) or accumulated_shots/device_max_experiments/device_max_shots>1/10:
                    saturated_shot = accumulated_shots
                    break
            counter += 1
        ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None,force_prob=True)
        ground_truth = dict_to_array(distribution_dict=ground_truth,force_prob=True)
        saturated_shot = max(min_saturated_shots,saturated_shot)
        qasm_evaluator_info['num_shots'] = saturated_shot
        saturated_prob = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info,force_prob=True)
        saturated_prob = dict_to_array(distribution_dict=saturated_prob,force_prob=True)
        saturated_shots.append(saturated_shot)
        ground_truths.append(ground_truth)
        saturated_probs.append(saturated_prob)
        saturated_chi2 = chi2_distance(target=ground_truth,obs=saturated_prob)
        # saturated_chi2 = wasserstein_distance(u_values=ground_truth,v_values=saturated_prob)
        print('%d qubit circuit saturated shots = %d, \u03C7^2 = %.3e'%(full_circ_size,saturated_shot,saturated_chi2))
        
    return saturated_shots, ground_truths, saturated_probs

def distribute_cluster_shots(total_shots,clusters,complete_path_map):
    cluster_shots = []
    for cluster_idx, cluster_circ in enumerate(clusters):
        O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
        num_instances = np.power(6,len(rho_qubits))*np.power(3,len(O_qubits))
        cluster_shots.append(math.ceil(total_shots/num_instances))
    return cluster_shots

def apply_measurement(circ):
    c = ClassicalRegister(len(circ.qubits), 'c')
    meas = QuantumCircuit(circ.qregs[0], c)
    meas.barrier(circ.qubits)
    meas.measure(circ.qubits,c)
    qc = circ+meas
    return qc

def combine_dict(dict_a, dict_sum):
    for key in dict_a:
        if key in dict_sum:
            dict_sum[key] = dict_sum[key]+dict_a[key]
        else:
            dict_sum[key] = dict_a[key]
    return dict_sum

def evaluate_circ(circ, backend, evaluator_info, force_prob):
    if backend == 'statevector_simulator':
        # print('using statevector simulator')
        backend = Aer.get_backend('statevector_simulator')
        backend_options = {'max_parallel_threads':1}
        job = execute(circ, backend=backend,backend_options=backend_options)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_dict = list_to_dict(l=outputstate)
        if force_prob:
            for key in outputstate_dict:
                x = outputstate_dict[key]
                outputstate_dict[key] = np.absolute(x)**2
        return outputstate_dict
    elif backend == 'noiseless_qasm_simulator':
        # print('using noiseless qasm simulator %d shots'%num_shots)
        backend = Aer.get_backend('qasm_simulator')
        backend_options = {'max_parallel_threads':1}
        qc = apply_measurement(circ)

        num_shots = evaluator_info['num_shots']
        noiseless_qasm_result = execute(qc, backend, shots=num_shots,backend_options=backend_options).result()
        
        noiseless_counts = noiseless_qasm_result.get_counts(0)
        assert sum(noiseless_counts.values())==num_shots
        if force_prob:
            for key in noiseless_counts:
                noiseless_counts[key] = noiseless_counts[key]/num_shots
        return noiseless_counts
    elif backend == 'noisy_qasm_simulator':
        # print('using noisy qasm simulator {} shots'.format(num_shots))
        backend = Aer.get_backend('qasm_simulator')
        backend_options = {'max_parallel_threads':1}
        qc=apply_measurement(circ)
        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'], 
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])
        
        num_shots = evaluator_info['num_shots']
        noisy_qasm_result = execute(experiments=mapped_circuit,
        backend=backend,
        noise_model=evaluator_info['noise_model'],
        coupling_map=evaluator_info['coupling_map'],
        basis_gates=evaluator_info['basis_gates'],
        shots=num_shots,backend_options=backend_options).result()

        noisy_counts = noisy_qasm_result.get_counts(0)
        assert sum(noisy_counts.values())==num_shots
        if force_prob:
            for key in noisy_counts:
                noisy_counts[key] = noisy_counts[key]/num_shots
        return noisy_counts
    else:
        raise Exception('Illegal backend :',backend)

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

def find_cluster_O_rho_qubit_positions(O_rho_pairs, cluster_circs):
    cluster_O_qubit_positions = {}
    cluster_rho_qubit_positions = {}
    for pair in O_rho_pairs:
        O_qubit, rho_qubit = pair
        O_cluster_idx, O_qubit_idx = O_qubit
        rho_cluster_idx, rho_qubit_idx = rho_qubit
        if O_cluster_idx not in cluster_O_qubit_positions:
            cluster_O_qubit_positions[O_cluster_idx] = [O_qubit_idx]
        else:
            cluster_O_qubit_positions[O_cluster_idx].append(O_qubit_idx)
        if rho_cluster_idx not in cluster_rho_qubit_positions:
            cluster_rho_qubit_positions[rho_cluster_idx] = [rho_qubit_idx]
        else:
            cluster_rho_qubit_positions[rho_cluster_idx].append(rho_qubit_idx)
    for cluster_idx in range(len(cluster_circs)):
        if cluster_idx not in cluster_O_qubit_positions:
            cluster_O_qubit_positions[cluster_idx] = []
        if cluster_idx not in cluster_rho_qubit_positions:
            cluster_rho_qubit_positions[cluster_idx] = []
    return cluster_O_qubit_positions, cluster_rho_qubit_positions

def find_cuts_pairs(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr+1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs

def effective_full_state_corresppndence(O_rho_pairs,cluster_circs):
    correspondence_map = {}
    for cluster_idx,circ in enumerate(cluster_circs):
        cluster_O_qubits = []
        total_num_qubits = len(circ.qubits)
        for pair in O_rho_pairs:
            O_qubit, _ = pair
            if O_qubit[0] == cluster_idx:
                cluster_O_qubits.append(O_qubit[1])
        effective_num_qubits = total_num_qubits - len(cluster_O_qubits)
        # print('cluster O qubits:',cluster_O_qubits)
        if effective_num_qubits>0:
            effective_states = itertools.product(range(2),repeat=effective_num_qubits)
            O_qubit_states = list(itertools.product(range(2),repeat=len(cluster_O_qubits)))
            cluster_correspondence = {}
            for effective_state in effective_states:
                # print('effective state:',effective_state)
                effective_state_index = int("".join(str(x) for x in effective_state), 2)
                corresponding_full_states = []
                for O_qubit_state in O_qubit_states:
                    full_state = list(effective_state)
                    for p,i in zip(cluster_O_qubits,O_qubit_state):
                        full_state.insert(p,i)
                    # print('O qubit state: {} --> full state: {}'.format(O_qubit_state,full_state))
                    full_state_index = int("".join(str(x) for x in full_state), 2)
                    corresponding_full_states.append(full_state_index)
                cluster_correspondence[effective_state_index] = corresponding_full_states
            correspondence_map[cluster_idx] = cluster_correspondence
        else:
            correspondence_map[cluster_idx] = None
    # print(correspondence_map)
    return correspondence_map

def smart_cluster_order(O_rho_pairs, cluster_circs):
    cluster_O_qubit_positions, cluster_rho_qubit_positions = find_cluster_O_rho_qubit_positions(O_rho_pairs, cluster_circs)
    smart_order = []
    cluster_Orho_qubits = []
    for cluster_idx in cluster_O_qubit_positions:
        num_O = len(cluster_O_qubit_positions[cluster_idx])
        num_rho = len(cluster_rho_qubit_positions[cluster_idx])
        cluster_Orho_qubits.append(num_O + num_rho)
        smart_order.append(cluster_idx)
        # print('Cluster %d has %d rho %d O'%(cluster_idx,num_O,num_rho))
    cluster_Orho_qubits, smart_order = zip(*sorted(zip(cluster_Orho_qubits, smart_order)))
    # print('smart order is:',smart_order)
    return smart_order

def find_inits_meas(cluster_circs, O_rho_pairs, s):
    # print('find initializations, measurement basis for:',s)
    clean_inits = []
    clean_meas = []
    for circ in cluster_circs:
        cluster_init = ['zero' for q in circ.qubits]
        cluster_meas = ['I' for q in circ.qubits]
        clean_inits.append(cluster_init)
        clean_meas.append(cluster_meas)
    
    clusters_init_meas = []
    cluster_meas = clean_meas
    cluster_inits = clean_inits
    for pair, s_i in zip(O_rho_pairs,s):
        O_qubit, rho_qubit = pair
        cluster_meas[O_qubit[0]][O_qubit[1]] = s_i
        cluster_inits[rho_qubit[0]][rho_qubit[1]] = s_i
    # print('inits:',cluster_inits)
    for i,m in zip(cluster_inits,cluster_meas):
        clusters_init_meas.append((tuple(i),tuple(m)))
    return tuple(clusters_init_meas)