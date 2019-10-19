from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.compiler import transpile
from qiskit.providers.aer import noise
import pickle
import itertools
import copy
import numpy as np
import progressbar as pb
from time import time
from mpi4py import MPI
import argparse
from helper_fun import simulate_circ, find_saturated_shots
import datetime as dt

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

def evaluate_cluster(complete_path_map, cluster_circ, combinations, backend, num_shots=None, provider=None):
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
        if backend!='statevector_simulator':
            device = provider.get_backend('ibmq_16_melbourne')
            properties = device.properties(dt.datetime(day=16, month=10, year=2019, hour=20))
            coupling_map = device.configuration().coupling_map
            noise_model = noise.device.basic_device_noise_model(properties)
            basis_gates = noise_model.basis_gates
            qasm_info = [device,properties,coupling_map,noise_model,basis_gates,num_shots]
            cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,backend=backend,qasm_info=qasm_info)
        else:
            cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,backend=backend,qasm_info=None)
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
    parser.add_argument('--saturated-shots',action="store_true",help='run saturated number of cluster shots')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    dimension,num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map = pickle.load(open(args.input_file, 'rb' ))
    provider = IBMQ.load_account()

    if rank == size-1:
        total_classical_time = 0
        total_quantum_time = 0
        all_cluster_prob = {}
        for cluster_idx,x in enumerate(clusters):
            all_cluster_prob[cluster_idx] = {}
        for i in range(num_workers):
            state = MPI.Status()
            rank_results,classical_time,quantum_time = comm.recv(source=MPI.ANY_SOURCE,status=state)
            total_classical_time += classical_time
            total_quantum_time += quantum_time
            for cluster_idx in range(len(all_cluster_prob)):
                all_cluster_prob[cluster_idx].update(rank_results[cluster_idx])
        if total_classical_time>0 and total_quantum_time>0:
            filename = args.input_file.replace('evaluator_input','hybrid_uniter_input')
        elif total_classical_time == 0 and total_quantum_time >0:
            filename = args.input_file.replace('evaluator_input','quantum_uniter_input')
        elif total_classical_time >0 and total_quantum_time == 0:
            filename = args.input_file.replace('evaluator_input','classical_uniter_input')
        else:
            raise Exception('evaluator time not recorded properly')
        if args.saturated_shots:
            filename = filename[:-2]+'_saturated.p'
        else:
            filename = filename[:-2]+'_sametotal.p'
        pickle.dump([num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map,all_cluster_prob,total_classical_time,total_quantum_time], open('%s'%filename,'wb'))
    else:
        rank_combinations = find_rank_combinations(clusters,complete_path_map,rank,size)
        rank_results = {}
        classical_time = 0
        quantum_time = 0
        for cluster_idx,cluster_combination in enumerate(rank_combinations):
            # NOTE: toggle here to control classical vs quantum evaluators
            if True:
            # if len(clusters[cluster_idx].qubits)<=5:
            # if False:
                print('rank %d runs %d combinations for cluster %d in classical evaluator'%(rank,len(cluster_combination),cluster_idx))
                classical_evaluator_begin = time()
                cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                cluster_circ=clusters[cluster_idx],
                combinations=cluster_combination,
                backend='statevector_simulator')
                classical_time += time()-classical_evaluator_begin
                rank_results[cluster_idx] = cluster_prob
            else:
                quantum_evaluator_begin = time()
                if args.saturated_shots:
                    rank_shots = find_saturated_shots(clusters[cluster_idx])
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=cluster_combination,
                    backend='noisy_qasm_simulator',num_shots=rank_shots,provider=provider)
                else:
                    rank_shots = max(int(num_shots/len(cluster_combination)/num_workers)+1,500)
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=cluster_combination,
                    backend='noisy_qasm_simulator',num_shots=rank_shots,provider=provider)
                print('rank %d runs %d combinations for cluster %d in quantum evaluator, %d shots'%
                (rank,len(cluster_combination),cluster_idx,rank_shots))
                quantum_time += time()-quantum_evaluator_begin
                rank_results[cluster_idx] = cluster_prob
        comm.send((rank_results,classical_time,quantum_time), dest=size-1)