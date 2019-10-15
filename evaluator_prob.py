from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
import pickle
import itertools
import copy
import os
import numpy as np
import progressbar as pb
from time import time
from mpi4py import MPI
import argparse

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def simulate_circ(circ, backend, noisy, qasm_info):
    if backend == 'statevector_simulator':
        # print('using statevector simulator')
        if noisy:
            raise Exception('statevector simulator does not run noisy evaluations')
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_ordered = [0 for sv in outputstate]
        for i, sv in enumerate(outputstate):
            reverse_i = reverseBits(i,len(circ.qubits))
            outputstate_ordered[reverse_i] = sv
        output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return output_prob
        # return outputstate_ordered
    elif backend == 'qasm_simulator':
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        backend = Aer.get_backend('qasm_simulator')
        if noisy:
            noise_model,coupling_map,basis_gates,num_shots,initial_layout = qasm_info
            # print('using noisy qasm simulator {} shots, NA = {}'.format(num_shots,initial_layout!=None))
            na_result = execute(experiments=qc,
            backend=backend,
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            shots=num_shots,
            initial_layout=initial_layout).result()
            na_counts = na_result.get_counts(qc)
            na_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
            for state in na_counts:
                reversed_state = reverseBits(int(state,2),len(circ.qubits))
                na_prob[reversed_state] = na_counts[state]/num_shots
            return na_prob
        else:
            _,_,_,num_shots,_ = qasm_info
            # print('using noiseless qasm simulator %d shots'%num_shots)
            job_sim = execute(qc, backend, shots=num_shots)
            result = job_sim.result()
            noiseless_counts = result.get_counts(qc)
            noiseless_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
            for state in noiseless_counts:
                reversed_state = reverseBits(int(state,2),len(circ.qubits))
                noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
            return noiseless_prob
    else:
        raise Exception('Illegal simulator:',backend)

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

def find_all_simulation_combinations(O_qubits, rho_qubits, num_qubits):
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','minus','plus_i','minus_i']
    # print('Rho qubits:',rho_qubits)
    all_inits = list(itertools.product(init_states,repeat=len(rho_qubits)))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(num_qubits)]
        for i in range(len(init)):
            complete_init[rho_qubits[i][1]] = init[i]
        complete_inits.append(complete_init)
    # print('initializations:',complete_inits)

    # print('O qubits:',O_qubits)
    all_meas = list(itertools.product(measurement_basis,repeat=len(O_qubits)))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['I' for i in range(num_qubits)]
        for i in range(len(meas)):
            complete_m[O_qubits[i][1]] = meas[i]
        complete_meas.append(complete_m)
    # print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    return combinations

def evaluate_cluster(complete_path_map, cluster_circ, combinations, backend, noisy, num_shots):
    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    properties = device.properties()
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties)
    basis_gates = noise_model.basis_gates

    cluster_prob = {}
    for _, combination in enumerate(combinations):
        cluster_dag = circuit_to_dag(cluster_circ)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = cluster_circ.qubits[i]
            if x == 'zero':
                continue
            elif x == 'one':
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal initialization : ',x)
        for i,x in enumerate(meas):
            q = cluster_circ.qubits[i]
            if x == 'I':
                continue
            elif x == 'X':
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                cluster_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        cluster_circ_inst = dag_to_circuit(cluster_dag)
        # print(inits, meas)
        # print(cluster_circ_inst)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(cluster_dag)
        initial_layout = noise_mapper.property_set['layout']
        qasm_info = [noise_model,coupling_map,basis_gates,num_shots,initial_layout] if backend=='qasm_simulator' else None
        cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,
        backend=backend,
        noisy=noisy,qasm_info=qasm_info)
        cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
    return cluster_prob

def find_rank_combinations(clusters,complete_path_map,rank,size):
    rank_combinations = []
    num_workers = size - 1
    for cluster_idx, cluster_circ in enumerate(clusters):
        O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
        combinations = find_all_simulation_combinations(O_qubits, rho_qubits, len(cluster_circ.qubits))
        count = int(len(combinations)/num_workers)
        remainder = len(combinations) % num_workers
        if rank<remainder:
            combinations_start = rank * (count + 1)
            combinations_stop = combinations_start + count + 1
        else:
            combinations_start = rank * count + remainder
            combinations_stop = combinations_start + (count - 1) + 1
        rank_combinations.append(combinations[combinations_start:combinations_stop])
    return rank_combinations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--input-file', metavar='S', type=str,help='which evaluator input file to run')
    parser.add_argument('--shots', metavar='N', type=str,help='number of shots of quantum evaluator')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    _,searcher_time,circ,fc_evaluations,clusters,complete_path_map = pickle.load(open(args.input_file, 'rb' ) )

    if rank == size-1:
        # print('MPI evaluator on cluster %d'%args.cluster_idx)
        all_cluster_prob = {}
        for cluster_idx,x in enumerate(clusters):
            all_cluster_prob[cluster_idx] = {}
        for i in range(num_workers):
            state = MPI.Status()
            rank_results,classical_time,quantum_time = comm.recv(source=MPI.ANY_SOURCE,status=state)
            for cluster_idx in range(len(all_cluster_prob)):
                all_cluster_prob[cluster_idx].update(rank_results[cluster_idx])
        filename = args.input_file.replace('evaluator_input','uniter_input')
        pickle.dump([complete_path_map, circ, clusters, all_cluster_prob, fc_evaluations, searcher_time, classical_time, quantum_time], open('%s'%filename,'wb'))
    else:
        rank_combinations = find_rank_combinations(clusters,complete_path_map,rank,size)
        rank_results = {}
        classical_time = 0
        quantum_time = 0
        for cluster_idx,cluster_combination in enumerate(rank_combinations):
            # if cluster_idx < int(len(clusters)/2):
            if True:
                print('rank %d runs %d combinations for cluster %d in classical evaluator'%(rank,len(cluster_combination),cluster_idx))
                classical_evaluator_begin = time()
                cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                cluster_circ=clusters[cluster_idx],
                combinations=cluster_combination,
                backend='statevector_simulator',noisy=False,num_shots=None)
                classical_time += time()-classical_evaluator_begin
                rank_results[cluster_idx] = cluster_prob
            else:
                print('rank %d runs %d combinations for cluster %d in quantum evaluator'%(rank,len(cluster_combination),cluster_idx))
                quantum_evaluator_begin = time()
                cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                cluster_circ=clusters[cluster_idx],
                combinations=cluster_combination,
                backend='qasm_simulator',noisy=True,num_shots=int(args.shots))
                quantum_time += time()-quantum_evaluator_begin
                rank_results[cluster_idx] = cluster_prob
        
        comm.send((rank_results,classical_time,quantum_time), dest=size-1)