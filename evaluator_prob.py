from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
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
from helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots
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

def evaluate_cluster(complete_path_map, cluster_circ, combinations, backend, evaluator_info):
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
        if backend=='statevector_simulator':
            cluster_inst_prob = evaluate_circ(circ=cluster_circ_inst,backend=backend,evaluator_info=None)
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
        elif backend=='noisy_qasm_simulator':
            cluster_inst_prob = evaluate_circ(circ=cluster_circ_inst,backend=backend,evaluator_info=evaluator_info)
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
        elif backend=='hardware':
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_circ_inst
        else:
            raise Exception('Illegal backend:',backend)
    return cluster_prob

def find_rank_combinations(evaluator_input,rank,size):
    num_workers = size - 1
    rank_combinations = {}
    for case in evaluator_input:
        rank_combinations[case] = []
        clusters = evaluator_input[case]['clusters']
        complete_path_map = evaluator_input[case]['complete_path_map']
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
            rank_combinations[case].append(combinations[combinations_start:combinations_stop])
    return rank_combinations

def get_filename(input_file,saturated_shots,evaluation_method):
    filename = None
    if evaluation_method == 'statevector_simulator':
        filename = input_file.replace('evaluator_input','classical_uniter_input')
    elif evaluation_method == 'noisy_qasm_simulator':
        filename = input_file.replace('evaluator_input','quantum_uniter_input')
    elif evaluation_method == 'hardware':
        filename = input_file.replace('evaluator_input','job_submittor_input')
    else:
        raise Exception('Illegal evaluation method :',evaluation_method)
    if evaluation_method != 'statevector_simulator' and saturated_shots:
        filename = filename[:-2]+'_saturated.p'
    elif evaluation_method != 'statevector_simulator' and not saturated_shots:
        filename = filename[:-2]+'_sametotal.p'
    elif evaluation_method == 'statevector_simulator':
        filename = filename
    else:
        raise Exception('Illegal combination :{}, saturated_shots = {}'.format(evaluation_method,saturated_shots))
    assert filename != None
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device input file to run')
    parser.add_argument('--circuit-name', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend to use')
    parser.add_argument('--saturated-shots',action="store_true",help='run saturated number of cluster shots')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    input_file = './benchmark_data/evaluator_input_{}_{}.p'.format(args.device_name,args.circuit_name)
    evaluator_input = pickle.load(open(input_file, 'rb' ))
    device_name = args.device_name

    if rank == size-1:
        evaluator_output = {}
        for case in evaluator_input:
            # case_dict = {'full_circ':full_circ,'fc_evaluations':fc_evaluations,'total_shots':total_shots,
            #     'searcher_time':searcher_time,'clusters':clusters,'complete_path_map':complete_path_map}
            evaluator_output[case] = copy.deepcopy(evaluator_input[case])
            evaluator_output[case]['classical_time'] = 0
            evaluator_output[case]['quantum_time'] = 0
            evaluator_output[case]['all_cluster_prob'] = {}
        for i in range(num_workers):
            state = MPI.Status()
            rank_results, rank_classical_time, rank_quantum_time = comm.recv(source=MPI.ANY_SOURCE,status=state)
            for case in evaluator_output:
                evaluator_output[case]['classical_time'] = max(evaluator_output[case]['classical_time'],rank_classical_time[case])
                evaluator_output[case]['quantum_time'] = max(evaluator_output[case]['quantum_time'],rank_quantum_time[case])
                if evaluator_output[case]['all_cluster_prob'] == {}:
                    evaluator_output[case]['all_cluster_prob'].update(rank_results[case])
                else:
                    for cluster_idx in evaluator_output[case]['all_cluster_prob']:
                        evaluator_output[case]['all_cluster_prob'][cluster_idx].update(rank_results[case][cluster_idx])
        filename = get_filename(input_file=input_file,saturated_shots=args.saturated_shots,evaluation_method=args.evaluation_method)
        pickle.dump(evaluator_output, open('%s'%filename,'wb'))
        print('-'*100)
    else:
        rank_combinations = find_rank_combinations(evaluator_input,rank,size)
        rank_results = {}
        rank_classical_time = {}
        rank_quantum_time = {}
        for case in rank_combinations:
            rank_results[case] = {}
            rank_quantum_time[case] = 0
            rank_classical_time[case] = 0
            clusters = evaluator_input[case]['clusters']
            complete_path_map = evaluator_input[case]['complete_path_map']
            total_shots = evaluator_input[case]['total_shots']
            for cluster_idx in range(len(rank_combinations[case])):
                if len(rank_combinations[case][cluster_idx]) > 0:
                    if args.evaluation_method == 'statevector_simulator':
                        classical_evaluator_begin = time()
                        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                        cluster_circ=clusters[cluster_idx],
                        combinations=rank_combinations[case][cluster_idx],
                        backend='statevector_simulator',evaluator_info=None)
                        elapsed_time = time()-classical_evaluator_begin
                        rank_classical_time[case] += elapsed_time
                        print('rank {} runs case {}, cluster_{} {}_qubits * {}_instances on CLASSICAL, classical time = {:.3e}'.format(
                            rank,case,cluster_idx,len(clusters[cluster_idx].qubits),
                            len(rank_combinations[case][cluster_idx]),elapsed_time))
                    elif args.evaluation_method == 'noisy_qasm_simulator':
                        evaluator_info = get_evaluator_info(circ=clusters[cluster_idx],device_name=device_name,
                        fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
                        quantum_evaluator_begin = time()
                        if args.saturated_shots:
                            evaluator_info['num_shots'] = get_circ_saturated_shots(circ=clusters[cluster_idx],accuracy=1e-1)
                        else:
                            evaluator_info['num_shots'] = int(total_shots/len(rank_combinations[case][cluster_idx])/num_workers)+1
                        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                        cluster_circ=clusters[cluster_idx],
                        combinations=rank_combinations[case][cluster_idx],
                        backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
                        elapsed_time = time()-quantum_evaluator_begin
                        rank_quantum_time[case] += elapsed_time
                        print('rank {} runs case {}, cluster_{} {}_qubits * {}_instances on {} QUANTUM SIMULATOR, {} shots = {}, quantum time  = {:.3e}'.format(
                                rank,case,cluster_idx,len(clusters[cluster_idx].qubits),
                                len(rank_combinations[case][cluster_idx]),device_name,'saturated' if args.saturated_shots else 'same_total',evaluator_info['num_shots'], elapsed_time))
                    elif args.evaluation_method == 'hardware':
                        quantum_evaluator_begin = time()
                        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                        cluster_circ=clusters[cluster_idx],
                        combinations=rank_combinations[case][cluster_idx],
                        backend='hardware',evaluator_info=None)
                        elapsed_time = time()-quantum_evaluator_begin
                        rank_quantum_time[case] += elapsed_time
                        print('case {}, cluster_{} {}_qubits * {}_instances on {} QUANTUM HARDWARE, {} shots'.format(
                                case,cluster_idx,len(clusters[cluster_idx].qubits),
                                len(rank_combinations[case][cluster_idx]),device_name,'saturated' if args.saturated_shots else 'same_total'))
                    else:
                        raise Exception('Illegal evaluation method:',args.evaluation_method)
                    rank_results[case][cluster_idx] = cluster_prob
                else:
                    rank_results[case][cluster_idx] = {}
        comm.send((rank_results,rank_classical_time,rank_quantum_time), dest=size-1)