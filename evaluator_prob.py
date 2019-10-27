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
from helper_fun import simulate_circ, find_saturated_shots, load_IBMQ, readout_mitigation
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
    if 'noisy' in backend:
        device = provider.get_backend('ibmq_16_melbourne')
        properties = device.properties(dt.datetime(day=16, month=10, year=2019, hour=20))
        coupling_map = device.configuration().coupling_map
        noise_model = noise.device.basic_device_noise_model(properties)
        basis_gates = noise_model.basis_gates
        dag = circuit_to_dag(cluster_circ)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(dag)
        initial_layout = noise_mapper.property_set['layout']
        meas_filter = readout_mitigation(circ=cluster_circ,initial_layout=initial_layout,num_shots=num_shots)
        qasm_info = {'device':device,
        'properties':properties,
        'coupling_map':coupling_map,
        'noise_model':noise_model,
        'basis_gates':basis_gates,
        'num_shots':num_shots,
        'meas_filter':meas_filter,
        'initial_layout':initial_layout}
    elif 'qasm' in backend:
        qasm_info = {'num_shots':num_shots}
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
            cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,backend=backend,qasm_info=qasm_info)
        else:
            cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,backend=backend,qasm_info=None)
        cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
    return cluster_prob

def find_rank_combinations(evaluator_input,rank,size):
    num_workers = size - 1
    rank_combinations = {}
    for key in evaluator_input:
        rank_combinations[key] = []
        _,_,_,_,_,clusters,complete_path_map = evaluator_input[key]
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
            rank_combinations[key].append(combinations[combinations_start:combinations_stop])
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

    evaluator_input = pickle.load(open(args.input_file, 'rb' ))
    provider = load_IBMQ()

    if rank == size-1:
        evaluator_output = {}
        for key in evaluator_input:
            dimension,num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map = evaluator_input[key]
            evaluator_output[key] = {}
            evaluator_output[key]['num_shots'] = num_shots
            evaluator_output[key]['circ'] = circ
            evaluator_output[key]['clusters'] = clusters
            evaluator_output[key]['searcher_time'] = searcher_time
            evaluator_output[key]['classical_time'] = 0
            evaluator_output[key]['quantum_time'] = 0
            evaluator_output[key]['complete_path_map'] = complete_path_map
            evaluator_output[key]['all_cluster_prob'] = {}
            evaluator_output[key]['fc_evaluations'] = fc_evaluations
        for i in range(num_workers):
            state = MPI.Status()
            rank_results, rank_classical_time, rank_quantum_time = comm.recv(source=MPI.ANY_SOURCE,status=state)
            for key in evaluator_output:
                evaluator_output[key]['classical_time'] += rank_classical_time[key]
                evaluator_output[key]['quantum_time'] += rank_quantum_time[key]
                if evaluator_output[key]['all_cluster_prob'] == {}:
                    evaluator_output[key]['all_cluster_prob'].update(rank_results[key])
                else:
                    for cluster_idx in evaluator_output[key]['all_cluster_prob']:
                        evaluator_output[key]['all_cluster_prob'][cluster_idx].update(rank_results[key][cluster_idx])
        quantum_eval = sum([evaluator_output[key]['quantum_time'] for key in evaluator_output])>0
        classical_eval = sum([evaluator_output[key]['classical_time'] for key in evaluator_output])>0
        if quantum_eval and classical_eval:
            filename = args.input_file.replace('evaluator_input','hybrid_uniter_input')
        elif not quantum_eval and classical_eval:
            filename = args.input_file.replace('evaluator_input','classical_uniter_input')
        elif quantum_eval and not classical_eval:
            filename = args.input_file.replace('evaluator_input','quantum_uniter_input')
        else:
            raise Exception('evaluator time not recorded properly')
        if args.saturated_shots:
            filename = filename[:-2]+'_saturated.p'
        else:
            filename = filename[:-2]+'_sametotal.p'
        pickle.dump(evaluator_output, open('%s'%filename,'wb'))
    else:
        rank_combinations = find_rank_combinations(evaluator_input,rank,size)
        rank_results = {}
        rank_classical_time = {}
        rank_quantum_time = {}
        for key in rank_combinations:
            rank_results[key] = {}
            rank_quantum_time[key] = 0
            rank_classical_time[key] = 0
            dimension,num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map = evaluator_input[key]
            for cluster_idx in range(len(rank_combinations[key])):
                # NOTE: toggle here for which evaluator to use
                # if True:
                if False:
                    print('rank {} runs case {}, cluster_{} * {} on CLASSICAL'.format(
                        rank,key,cluster_idx,
                        len(rank_combinations[key][cluster_idx])))
                    classical_evaluator_begin = time()
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=rank_combinations[key][cluster_idx],
                    backend='statevector_simulator')
                    rank_classical_time[key] += time()-classical_evaluator_begin
                elif args.saturated_shots:
                    quantum_evaluator_begin = time()
                    rank_shots = find_saturated_shots(clusters[cluster_idx])
                    print('rank {} runs case {}, cluster_{} * {} on QUANTUM, saturated shots = {}'.format(
                        rank,key,cluster_idx,
                        len(rank_combinations[key][cluster_idx]),rank_shots))
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=rank_combinations[key][cluster_idx],
                    backend='noisy_qasm_simulator',num_shots=rank_shots,provider=provider)
                    rank_quantum_time[key] += time()-quantum_evaluator_begin
                else:
                    quantum_evaluator_begin = time()
                    rank_shots = max(int(num_shots/len(rank_combinations[key][cluster_idx])/num_workers)+1,500)
                    print('rank {} runs case {}, cluster_{} * {} on QUANTUM, sameTotal shots = {}'.format(
                        rank,key,cluster_idx,
                        len(rank_combinations[key][cluster_idx]),rank_shots))
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=rank_combinations[key][cluster_idx],
                    backend='noisy_qasm_simulator',num_shots=rank_shots,provider=provider)
                    rank_quantum_time[key] += time()-quantum_evaluator_begin
                rank_results[key][cluster_idx] = cluster_prob
            print('classical time = ',rank_classical_time[key], 'quantum time = ',rank_quantum_time[key])
        comm.send((rank_results,rank_classical_time,rank_quantum_time), dest=size-1)