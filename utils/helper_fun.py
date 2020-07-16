import os
import pickle
import numpy as np
import math

from qiskit import Aer, IBMQ, execute
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import NoiseAdaptiveLayout

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore, gen_adder
from utils.conversions import dict_to_array, memory_to_dict

def get_dirname(circuit_type,cc_size,full_circ_size,techniques,eval_mode,field):
    if field=='generator':
        dirname = './source_data/%s/cc_%d/q_%d'%(circuit_type,cc_size,full_circ_size)
    elif field=='evaluator':
        dirname = './source_data/%s/cc_%d/q_%d/%s/eval'%(circuit_type,cc_size,full_circ_size,eval_mode)
    elif field=='measure':
        dirname = './source_data/%s/cc_%d/q_%d/%s/measure'%(circuit_type,cc_size,full_circ_size,eval_mode)
    elif field=='rank':
        dirname = './processed_data/%s/cc_%d/q_%d/%s_%d_%d_%d_%d'%(circuit_type,cc_size,full_circ_size,eval_mode,*techniques)
    elif field=='slurm':
        dirname = './slurm/%s/cc_%d/q_%d/%s_%d_%d_%d_%d'%(circuit_type,cc_size,full_circ_size,eval_mode,*techniques)
    else:
        raise Exception('Illegal field = %s'%field)
    return dirname

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

def apply_measurement(circ):
    c = ClassicalRegister(len(circ.qubits), 'c')
    meas = QuantumCircuit(circ.qregs[0], c)
    meas.barrier(circ.qubits)
    meas.measure(circ.qubits,c)
    qc = circ+meas
    return qc

def evaluate_circ(circ,backend,device_name):
    backend_options = {'max_memory_mb': 2**30*16/1024**2}
    # NOTE: control shots here
    num_shots = max(1024,2**circ.n_qubits)
    if backend=='statevector_simulator':
        # print('using statevector simulator')
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend=backend, optimization_level=0)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate = [np.absolute(x)**2 for x in outputstate]
        outputstate = np.array(outputstate)
        return outputstate
    elif backend == 'noiseless_qasm_simulator':
        # print('using noiseless qasm simulator %d shots'%num_shots)
        backend = Aer.get_backend('qasm_simulator')
        qc = apply_measurement(circ)

        noiseless_qasm_result = execute(qc, backend, shots=num_shots,backend_options=backend_options).result()
        
        noiseless_counts = noiseless_qasm_result.get_counts(0)
        assert sum(noiseless_counts.values())==num_shots
        noiseless_counts = dict_to_array(distribution_dict=noiseless_counts,force_prob=True)
        return noiseless_counts
    elif backend == 'noisy_qasm_simulator':
        # print('using noisy qasm simulator {} shots'.format(num_shots))
        backend = Aer.get_backend('qasm_simulator')
        qc=apply_measurement(circ)

        evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,fields=
                        ['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])

        device = evaluator_info['device']
        basis_gates = evaluator_info['basis_gates']
        coupling_map = evaluator_info['coupling_map']
        properties = evaluator_info['properties']
        initial_layout = evaluator_info['initial_layout']
        noise_model = evaluator_info['noise_model']
        
        mapped_circuit = transpile(qc,backend=device,initial_layout=initial_layout)

        # print('evaluate_circ function:')
        # print(qc)
        # print(mapped_circuit)
        noisy_qasm_result = execute(experiments=mapped_circuit,
        backend=backend,noise_model=noise_model,
        shots=num_shots,backend_options=backend_options).result()

        noisy_counts = noisy_qasm_result.get_counts(0)
        assert sum(noisy_counts.values())==num_shots
        noisy_counts = dict_to_array(distribution_dict=noisy_counts,force_prob=True)
        return noisy_counts
    
    else:
        raise Exception('Backend %s illegal'%backend)

def load_IBMQ():
    token = '9056ff772ff2e0f19de847fc8980b6e0121b561832de7dfb72bb23b085c1dc4a62cde82392f7d74e655465a9d997dd970858a568434f1b97038e70bf44b6c8a6'
    if len(IBMQ.stored_account()) == 0:
        IBMQ.save_account(token)
        IBMQ.load_account()
    elif IBMQ.active_account() == None:
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-ornl', group='bes-qis', project='argonne')
    return provider

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n/val)
        if val*co_val == n:
            return val, co_val
        else:
            val -= 1

def generate_circ(full_circ_size,circuit_type):
    def gen_secret(num_qubit):
        num_digit = num_qubit-1
        num = 2**num_digit-1
        num = bin(num)[2:]
        num_with_zeros = str(num).zfill(num_digit)
        return num_with_zeros

    i,j = factor_int(full_circ_size)
    if circuit_type == 'supremacy_linear':
        full_circ = gen_supremacy(1,full_circ_size,8)
    elif circuit_type == 'supremacy_grid':
        if abs(i-j)<=2:
            full_circ = gen_supremacy(i,j,8)
        else:
            full_circ = QuantumCircuit()
    elif circuit_type == 'hwea':
        full_circ = gen_hwea(i*j,1)
    elif circuit_type == 'bv':
        full_circ = gen_BV(gen_secret(i*j),barriers=False)
    elif circuit_type == 'aqft':
        approximation_degree=int(math.log(full_circ_size,2)+2)
        # print('%d-qubit AQFT, approximation_degree = %d'%(full_circ_size,approximation_degree))
        full_circ = gen_qft(width=i*j, approximation_degree=approximation_degree,barriers=False)
    elif circuit_type == 'sycamore':
        full_circ = gen_sycamore(i,j,8)
    elif circuit_type == 'adder':
        full_circ = gen_adder(nbits=int((full_circ_size-2)/2),barriers=False)
    else:
        raise Exception('Illegal circuit type:',circuit_type)
    return full_circ

def get_evaluator_info(circ,device_name,fields):
    provider = load_IBMQ()
    device = provider.get_backend(device_name)
    properties = device.properties()
    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(properties)
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

def find_process_jobs(jobs,rank,num_workers):
    count = int(len(jobs)/num_workers)
    remainder = len(jobs) % num_workers
    if rank<remainder:
        jobs_start = rank * (count + 1)
        jobs_stop = jobs_start + count + 1
    else:
        jobs_start = rank * count + remainder
        jobs_stop = jobs_start + (count - 1) + 1
    process_jobs = list(jobs[jobs_start:jobs_stop])
    return process_jobs