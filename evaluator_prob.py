from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
import pickle
import glob
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

def simulate_circ(circ, simulator, noisy=False, provider_info=None, num_shots=1024,initial_layout=None):
    if noisy:
        if provider_info==None:
            raise Exception('Provider info is required for noisy evaluation')
        if simulator!='qasm_simulator' and simulator!='ibmq_qasm_simulator':
            raise Exception('Noisy evaluation cannot use {} evaluator'.format(simulator))
    else:
        if simulator == 'ibmq_qasm_simulator':
            raise Exception('Noiseless evaluation cannot use ibmq_qasm_simulator')
    if simulator == 'statevector_simulator':
        backend = Aer.get_backend(simulator)
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_ordered = [0 for sv in outputstate]
        for i, sv in enumerate(outputstate):
            reverse_i = reverseBits(i,len(circ.qubits))
            outputstate_ordered[reverse_i] = sv
        output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return output_prob
    elif simulator == 'qasm_simulator':
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        backend = Aer.get_backend(simulator)
        if not noisy:
            # print('noiseless qasm with %d shots'%num_shots)
            job_sim = execute(qc, backend, shots=num_shots)
            result = job_sim.result()
            counts = result.get_counts(qc)
        else:
            provider, noise_model, coupling_map, basis_gates = provider_info
            result = execute(qc, backend,
                       noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates,shots=num_shots,initial_layout=initial_layout).result()
            counts = result.get_counts(qc)
        prob_ordered = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            prob_ordered[reversed_state] = counts[state]/num_shots
        return prob_ordered
    elif simulator == 'ibmq_qasm_simulator':
        provider, noise_model, coupling_map, basis_gates = provider_info
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        backend = provider.get_backend('ibmq_qasm_simulator')
        result_noise = execute(qc, backend, 
                       noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates,
                       shots=num_shots).result()
        counts_noise = result_noise.get_counts(qc)
        prob_ordered = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in counts_noise:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            prob_ordered[reversed_state] = counts_noise[state]/num_shots
        return prob_ordered
    else:
        raise Exception('Illegal simulator:',simulator)

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

def evaluate_cluster(complete_path_map, cluster, combinations, provider_info=None, simulator_backend='statevector_simulator',noisy=False,num_shots=int(1e5)):
    cluster_prob = {}
    for counter, combination in enumerate(combinations):
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
        cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,
        simulator=simulator_backend,
        noisy=noisy,
        provider_info=provider_info,
        num_shots=num_shots)
        cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
    return cluster_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--cluster-idx', metavar='N', type=int,help='which cluster pickle file to run')
    parser.add_argument('--backend', metavar='S', type=str,help='which Qiskit backend')
    parser.add_argument('--noisy', action='store_true',help='noisy evaluation?')
    parser.add_argument('--dirname', metavar='S', type=str,default='./data',help='which directory?')
    parser.add_argument('--shots', metavar='N', type=int,default=int(1e5),help='number of shots')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    dirname = args.dirname
    clusters, complete_path_map, provider_info = pickle.load( open( '%s/evaluator_input.p'%dirname, 'rb' ) )

    cluster_circ = clusters[args.cluster_idx]
    O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,args.cluster_idx)
    combinations = find_all_simulation_combinations(O_qubits, rho_qubits, len(cluster_circ.qubits))

    count = int(len(combinations)/num_workers)
    remainder = len(combinations) % num_workers

    if rank == size-1:
        print('Evaluator master rank')
        cluster_prob = {}
        # bar = pb.ProgressBar(max_value=num_workers)
        for i in range(num_workers):
            state = MPI.Status()
            rank_cluster_prob = comm.recv(source=MPI.ANY_SOURCE,status=state)
            cluster_prob.update(rank_cluster_prob)
            # bar.update(i)
        pickle.dump(cluster_prob, open( '%s/cluster_%d_prob.p'%(dirname,args.cluster_idx), 'wb' ))
    elif rank<remainder:
        combinations_start = rank * (count + 1)
        combinations_stop = combinations_start + count + 1
        rank_combinations = combinations[combinations_start:combinations_stop]
        print('rank %d runs %d combinations for cluster %d'%(rank,len(rank_combinations),args.cluster_idx))
        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
        cluster=cluster_circ,
        combinations=rank_combinations,
        provider_info=provider_info,
        simulator_backend=args.backend,noisy=args.noisy,num_shots=args.shots)
        comm.send(cluster_prob, dest=size-1)
    else:
        combinations_start = rank * count + remainder
        combinations_stop = combinations_start + (count - 1) + 1
        rank_combinations = combinations[combinations_start:combinations_stop]
        print('rank %d runs %d combinations for cluster %d'%(rank,len(rank_combinations),args.cluster_idx))
        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
        cluster=cluster_circ,
        combinations=rank_combinations,
        provider_info=provider_info,
        simulator_backend=args.backend,noisy=args.noisy,num_shots=args.shots)
        comm.send(cluster_prob, dest=size-1)