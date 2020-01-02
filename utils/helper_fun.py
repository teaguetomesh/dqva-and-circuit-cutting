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
    saturated_probs = []
    ground_truths = []
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
                    saturated_probs.append(accumulated_prob)
                    ground_truths.append(ground_truth)
                    print('%d qubit circuit saturated shots = %d, \u0394H = %.3e'%(full_circ_size,accumulated_shots,ce_l[-2]))
                    break
            counter += 1
    return saturated_shots, saturated_probs, ground_truths

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

def dict_to_prob(distribution_dict,reverse=False):
    state = list(distribution_dict.keys())[0]
    num_qubits = len(state)
    num_shots = sum(distribution_dict.values())
    prob = [0 for x in range(np.power(2,num_qubits))]
    for state in distribution_dict:
        if reverse:
            reversed_state = reverseBits(int(state,2),num_qubits)
            prob[reversed_state] = distribution_dict[state]/num_shots
        else:
            prob[int(state,2)] = distribution_dict[state]/num_shots
    return prob

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
        sv_prob = [np.power(np.absolute(x),2) for x in outputstate]
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
        noiseless_prob = dict_to_prob(distribution_dict=noiseless_counts)
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
        noisy_prob = dict_to_prob(distribution_dict=noisy_counts)
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
            # job = evaluator_info['device'].run(qobj)
            job = Aer.get_backend('qasm_simulator').run(qobj)
            jobs.append({'job':job,'circ':circ,'mapped_circuit_l':circs_l,'evaluator_info':evaluator_info})
            print('Submitted {:d} reps * {:d} shots to hardware, job_id = {}'.format(s['reps'],s['shots'],job.job_id()))
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
    assert len(meas_calibs)<=device.configuration().max_experiments/3*2

    # Execute the calibration circuits
    meas_calibs_transpiled = transpile(meas_calibs, backend=device)
    qobj = assemble(meas_calibs_transpiled, backend=device, shots=num_shots)
    # job = device.run(qobj)
    job = Aer.get_backend('qasm_simulator').run(qobj)
    print('Submitted measurement filter, {:d} calibration circuits * {:d} shots, job_id = {}'.format(len(meas_calibs),num_shots,job.job_id()))
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