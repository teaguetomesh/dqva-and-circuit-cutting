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

def get_filename(experiment_name,circuit_type,device_name,field,evaluation_method=None,shots_mode=None):
    dirname = './{}/benchmark_data/{}_{}/'.format(experiment_name,circuit_type,device_name)
    if field == 'evaluator_input':
        evaluator_input_filename = 'evaluator_input_{}.p'.format(device_name)
        return dirname, evaluator_input_filename
    elif field == 'job_submittor_input':
        job_submittor_input_filename = 'job_submittor_input_{}.p'.format(device_name)
        return dirname, job_submittor_input_filename
    elif field == 'uniter_input':
        if evaluation_method == 'statevector_simulator':
            uniter_input_filename = 'classical_uniter_input_{}_'.format(device_name)
        elif evaluation_method == 'noisy_qasm_simulator':
            uniter_input_filename = 'quantum_uniter_input_{}_'.format(device_name)
        elif evaluation_method == 'hardware':
            uniter_input_filename = 'hw_uniter_input_{}_'.format(device_name)
        else:
            raise Exception('Illegal evaluation method :',evaluation_method)
        uniter_input_filename += shots_mode + '.p'
        return dirname, uniter_input_filename
    elif field == 'plotter_input':
        if evaluation_method == 'statevector_simulator':
            plotter_input_filename = 'classical_plotter_input_{}_'.format(device_name)
        elif evaluation_method == 'noisy_qasm_simulator':
            plotter_input_filename = 'quantum_plotter_input_{}_'.format(device_name)
        elif evaluation_method == 'hardware':
            plotter_input_filename = 'hw_plotter_input_{}_'.format(device_name)
        else:
            raise Exception('Illegal evaluation method :',evaluation_method)
        plotter_input_filename += shots_mode + '.p'
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

def cross_entropy(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    sum_of_prob = sum(obs)
    obs = [x/sum_of_prob for x in obs]
    assert abs(sum(obs)-1) < 1e-10
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            h += p*np.log(p/q)
    return h

def fidelity(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    sum_of_prob = sum(obs)
    obs = [x/sum_of_prob for x in obs]
    if abs(sum(obs)-1) > 1e-10:
        print('sum of obs =',sum(obs))
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

def get_circ_saturated_shots(circs,device_name):
    saturated_shots = []
    for circ_idx, circ in enumerate(circs):
        full_circ_size = len(circ.qubits)
        ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)
        shots_increment = 1024
        
        qasm_evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
        fields=['device','basis_gates','coupling_map','properties','initial_layout'])
        qasm_evaluator_info['num_shots'] = shots_increment
        device_max_shots = qasm_evaluator_info['device'].configuration().max_shots
        device_max_experiments = int(qasm_evaluator_info['device'].configuration().max_experiments/3*2)
        ce_l = []
        counter = 1
        accumulated_prob = [0 for i in range(np.power(2,len(circ.qubits)))]
        while 1:
            noiseless_prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info)
            accumulated_prob = [(x*(counter-1)+y)/counter for x,y in zip(accumulated_prob,noiseless_prob_batch)]
            assert abs(sum(accumulated_prob)-1)<1e-5
            accumulated_ce = cross_entropy(target=ground_truth,obs=accumulated_prob)
            ce_l.append(accumulated_ce)
            if len(ce_l)>=3:
                accumulated_shots = int((len(ce_l)-1)*shots_increment)
                first_derivative = (ce_l[-1]+ce_l[-3])/(2*shots_increment)
                second_derivative = (ce_l[-1]+ce_l[-3]-2*ce_l[-2])/(2*np.power(shots_increment,2))
                if (abs(first_derivative)<1e-6 and abs(second_derivative) < 1e-10) or accumulated_shots/device_max_experiments/device_max_shots>1:
                    saturated_shots.append(accumulated_shots)
                    print('%d qubit circuit saturated shots = %d, \u0394H = %.3e'%(full_circ_size,accumulated_shots,ce_l[-2]))
                    break
            counter += 1
    return saturated_shots

def distribute_cluster_shots(total_shots,clusters,complete_path_map):
    cluster_shots = []
    for cluster_idx, cluster_circ in enumerate(clusters):
        O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
        num_instances = np.power(6,len(rho_qubits))*np.power(3,len(O_qubits))
        cluster_shots.append(math.ceil(total_shots/num_instances))
    return cluster_shots

def schedule_job(circs,shots,max_experiments,max_shots):
    # print('%d circuits, %d shots each, %d max_experiments, %d max_shots'%(len(circs),shots,max_experiments,max_shots))
    shots = 1024*math.ceil(shots/1024)
    if len(circs)==0 or shots==0:
        return []
    elif len(circs)<=max_experiments and shots<=max_shots:
        current_schedule = {'circs':circs,'shots':shots,'reps':1}
        return [current_schedule]
    elif len(circs)>max_experiments and shots<=max_shots:
        curr_circs = {}
        next_circs = {}
        for init_meas in circs:
            if len(curr_circs)<max_experiments:
                curr_circs[init_meas] = circs[init_meas]
            else:
                next_circs[init_meas] = circs[init_meas]
        current_schedule = {'circs':curr_circs,'shots':shots,'reps':1}
        next_schedule = schedule_job(circs=next_circs,shots=shots,max_experiments=max_experiments,max_shots=max_shots)
        return [current_schedule]+next_schedule
    elif len(circs)<=max_experiments and shots>max_shots:
        batch_shots = max_shots
        while 1:
            if shots%batch_shots == 0:
                break
            else:
                batch_shots -= 1024
        shots_repetitions_required = int(shots/batch_shots)
        repetitions_allowed = int(max_experiments/len(circs))
        if shots_repetitions_required <= repetitions_allowed:
            reps = shots_repetitions_required
            remaining_shots = shots - reps*batch_shots
            assert remaining_shots==0
            current_schedule = {'circs':circs,'shots':batch_shots,'reps':reps}
            return [current_schedule]
        else:
            min_shots_repetitions_required = int(shots/max_shots)
            reps = min(repetitions_allowed,min_shots_repetitions_required)
            remaining_shots = shots - reps*max_shots
            current_schedule = {'circs':circs,'shots':max_shots,'reps':reps}
            next_schedule = schedule_job(circs=circs,shots=remaining_shots,max_experiments=max_experiments,max_shots=max_shots)
            return [current_schedule]+next_schedule
    elif len(circs)>max_experiments and shots>max_shots:
        left_circs = {}
        right_circs = {}
        for init_meas in circs:
            if len(left_circs) < max_experiments:
                left_circs[init_meas] = circs[init_meas]
            else:
                right_circs[init_meas] = circs[init_meas]
        left_schedule = schedule_job(circs=left_circs,shots=shots,max_experiments=max_experiments,max_shots=max_shots)
        right_schedule = schedule_job(circs=right_circs,shots=shots,max_experiments=max_experiments,max_shots=max_shots)
        return left_schedule + right_schedule
    else:
        raise Exception('This condition should not happen')

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

def combine_dict(dict_a, dict_sum):
    for key in dict_a:
        if key in dict_sum:
            dict_sum[key] = dict_sum[key]+dict_a[key]
        else:
            dict_sum[key] = dict_a[key]
    return dict_sum

def evaluate_circ(circ, backend, evaluator_info):
    if backend == 'statevector_simulator':
        # print('using statevector simulator')
        backend = Aer.get_backend('statevector_simulator')
        backend_options = {'max_parallel_threads':1}
        job = execute(circ, backend=backend,backend_options=backend_options)
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
        backend_options = {'max_parallel_threads':1}
        qc = apply_measurement(circ)

        num_shots = evaluator_info['num_shots']
        noiseless_qasm_result = execute(qc, backend, shots=num_shots,backend_options=backend_options).result()
        
        noiseless_counts = noiseless_qasm_result.get_counts(0)
        assert sum(noiseless_counts.values())>=num_shots
        noiseless_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noiseless_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
        return noiseless_prob
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
        assert sum(noisy_counts.values())>=num_shots
        noisy_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noisy_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noisy_prob[reversed_state] = noisy_counts[state]/num_shots
        return noisy_prob
    elif backend == 'hardware':
        jobs = []
        qc=apply_measurement(circ)

        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])

        device_max_shots = evaluator_info['device'].configuration().max_shots
        device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/3*2)

        schedule = schedule_job(circs={'fc':mapped_circuit},shots=evaluator_info['num_shots'],max_experiments=device_max_experiments,max_shots=device_max_shots)
        
        for s in schedule:
            circs_l = []
            for init_meas in s['circs']:
                reps_l = [s['circs'][init_meas] for i in range(s['reps'])]
                circs_l += reps_l
            qobj = assemble(circs_l, backend=evaluator_info['device'], shots=s['shots'])
            print('Submitted full circuit %d shots, %d reps to hardware'%(s['shots'],s['reps']))
            # job = evaluator_info['device'].run(qobj)
            job = Aer.get_backend('qasm_simulator').run(qobj)
            jobs.append({'job':job,'circ':circ,'mapped_circuit_l':circs_l,'evaluator_info':evaluator_info})
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
    print('Calculating measurement filter, %d-qubit calibration circuits * %d * %.3e shots.'%(len(meas_calibs[0].qubits),len(meas_calibs),num_shots))
    assert len(meas_calibs)<=device.configuration().max_experiments/3*2

    # Execute the calibration circuits
    meas_calibs_transpiled = transpile(meas_calibs, backend=device)
    qobj = assemble(meas_calibs_transpiled, backend=device, shots=num_shots)
    # job = device.run(qobj)
    job = Aer.get_backend('qasm_simulator').run(qobj)
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