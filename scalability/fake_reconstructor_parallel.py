import itertools
import numpy as np
import math
import pickle
import glob
from time import time
from scipy.stats import wasserstein_distance
import argparse
from utils.helper_fun import get_filename, read_file, find_cluster_O_rho_qubit_positions, find_cuts_pairs, find_inits_meas
from utils.metrics import chi2_distance
from utils.conversions import reverse_prob
from scalability.fake_reconstruct_helper_fun import fake_reconstruct
import copy
import os.path
from numba import jit, njit, prange
from mpi4py import MPI

def reconstructed_reorder(unordered,complete_path_map,smart_order,unordered_start,unordered_end):
    # print(complete_path_map)
    # print('ordering reconstructed sv')
    ordered = np.zeros(len(unordered))
    cluster_out_qubits = {}
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        # print('output qubit = ', output_qubit)
        if output_qubit[0] in cluster_out_qubits:
            cluster_out_qubits[output_qubit[0]].append((output_qubit[1],input_qubit.index))
        else:
            cluster_out_qubits[output_qubit[0]] = [(output_qubit[1],input_qubit.index)]
    # print(cluster_out_qubits)
    for cluster_idx in cluster_out_qubits:
        cluster_out_qubits[cluster_idx].sort()
        cluster_out_qubits[cluster_idx] = [x[1] for x in cluster_out_qubits[cluster_idx]]
    # print(cluster_out_qubits)
    unordered_qubit_idx = []
    for cluster_idx in smart_order:
        if cluster_idx in cluster_out_qubits:
            unordered_qubit_idx += cluster_out_qubits[cluster_idx]
    # print(unordered_qubit_idx)
    for idx, sv in enumerate(unordered[unordered_start:unordered_end]):
        idx += unordered_start
        bin_idx = bin(idx)[2:].zfill(len(unordered_qubit_idx))
        # print('sv bin_idx=',bin_idx)
        ordered_idx = [0 for i in unordered_qubit_idx]
        for jdx, i in enumerate(bin_idx):
            ordered_idx[unordered_qubit_idx[jdx]] = i
        # print(ordered_idx)
        ordered_idx = int("".join(str(x) for x in ordered_idx), 2)
        ordered[ordered_idx] = sv
        # print('unordered %d --> ordered %d'%(idx,ordered_idx),'sv=',sv)
    return ordered

def get_combinations(complete_path_map):
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    # print('O rho qubits pairs:',O_rho_pairs)

    basis = ['I','X','Y','Z']

    combinations = list(itertools.product(basis,repeat=len(O_rho_pairs)))
    return combinations

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

def find_rank_combinations(combinations,rank,num_workers):
    count = int(len(combinations)/num_workers)
    remainder = len(combinations) % num_workers
    if rank<remainder:
        combinations_start = rank * (count + 1)
        combinations_stop = combinations_start + count + 1
    else:
        combinations_start = rank * count + remainder
        combinations_stop = combinations_start + (count - 1) + 1
    # rank_combinations = combinations[combinations_start:combinations_stop]
    return combinations_start, combinations_stop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uniter')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    dirname, uniter_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,device_name='fake',field='uniter_input',evaluation_method='fake')
    uniter_input = read_file(dirname+uniter_input_filename)
    dirname, plotter_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,device_name='fake',field='plotter_input',evaluation_method='fake')
    plotter_input = read_file(dirname+plotter_input_filename)

    if rank == size-1:
        print('-'*50,'Reconstructor','-'*50,flush=True)
        print('Existing cases:',plotter_input.keys())
        counter = len(plotter_input.keys())
        for case in uniter_input:
            if case in plotter_input:
                continue
            print('case {}'.format(case),flush=True)
            case_dict = copy.deepcopy(uniter_input[case])
            print('Cut into ',[len(x.qubits) for x in case_dict['clusters']],'clusters')

            combinations = get_combinations(uniter_input[case]['complete_path_map'])
            reconstructed_prob = np.zeros(2**case[1])

            compute_begin = time()
            for i in range(num_workers):
                combinations_start, combinations_stop = find_rank_combinations(combinations,i,num_workers)
                rank_combinations = combinations[combinations_start:combinations_stop]
                comm.send((case,rank_combinations), dest=i)
            for i in range(num_workers):
                state = MPI.Status()
                rank_reconstructed_prob, smart_order = comm.recv(source=MPI.ANY_SOURCE,status=state)
                reconstructed_prob += rank_reconstructed_prob
            compute_time = time() - compute_begin
            print('Quantum took %.3f seconds'%case_dict['quantum_time'])
            print('Compute took %.3f seconds'%compute_time)
            
            reorder_begin = time()
            for i in range(num_workers):
                combinations_start, combinations_stop = find_rank_combinations(reconstructed_prob,i,num_workers)
                comm.send((reconstructed_prob,combinations_start,combinations_stop), dest=i)
            reconstructed_prob = np.zeros(2**case[1])
            for i in range(num_workers):
                state = MPI.Status()
                rank_reconstructed_prob = comm.recv(source=MPI.ANY_SOURCE,status=state)
                reconstructed_prob += rank_reconstructed_prob
            reorder_time = time() - reorder_begin
            print('Reorder took %.3f seconds'%reorder_time)
            
            reverse_begin = time()
            norm = sum(reconstructed_prob)
            reconstructed_prob = reconstructed_prob/norm
            reconstructed_prob = reverse_prob(prob_l=reconstructed_prob)
            reverse_time = time() - reverse_begin
            print('Reverse took %.3f seconds'%reverse_time)

            # print('reconstruction len =', len(reconstructed_prob),'probabilities sum = ', sum(reconstructed_prob))
            assert len(reconstructed_prob) == 2**case[1] and abs(sum(reconstructed_prob)-1)<1e-5

            hybrid_time = case_dict['searcher_time'] + case_dict['quantum_time'] + compute_time + reorder_time + reverse_time
            print('QC hybrid took %.3f seconds, classical took %.3f seconds'%(hybrid_time,case_dict['std_time']))

            # pickle.dump({case:case_dict}, open('%s'%(dirname+plotter_input_filename),'wb'))
            counter += 1
            print('Reconstruction output has %d cases'%counter,flush=True)
            print('-'*100)

        for i in range(num_workers):
            comm.send('DONE', dest=i)
    else:
        while 1:
            state = MPI.Status()
            rank_input = comm.recv(source=size-1,status=state)
            if rank_input == 'DONE':
                break
            else:
                case,rank_combinations = rank_input
                complete_path_map = uniter_input[case]['complete_path_map']
                full_circ = uniter_input[case]['full_circ']
                cluster_circs = uniter_input[case]['clusters']
                cluster_probs = uniter_input[case]['all_cluster_prob']
        
                get_terms_begin = time()
                reconstructed_prob, scaling_factor, smart_order = fake_reconstruct(complete_path_map=uniter_input[case]['complete_path_map'],
                combinations=rank_combinations,
                full_circ=uniter_input[case]['full_circ'], cluster_circs=uniter_input[case]['clusters'],
                cluster_sim_probs=uniter_input[case]['all_cluster_prob'])
                get_terms_time = time() - get_terms_begin
                #print('Rank %d reconstruction took %.3f seconds'%(rank,get_terms_time))

                reconstructed_prob = reconstructed_prob/scaling_factor

                comm.send((reconstructed_prob,smart_order), dest=size-1)

                state = MPI.Status()
                reconstructed_prob,combinations_start,combinations_stop = comm.recv(source=size-1,status=state)
                rank_reconstructed_prob = reconstructed_reorder(reconstructed_prob,complete_path_map=uniter_input[case]['complete_path_map'],smart_order=smart_order,
                unordered_start=combinations_start,unordered_end=combinations_stop)
                # print('Rank %d reordered %d-%d, len = %d, sum = %.2f'%(rank,combinations_start,combinations_stop,
                # len(rank_reconstructed_prob),sum(rank_reconstructed_prob)))
                comm.send(rank_reconstructed_prob, dest=size-1)