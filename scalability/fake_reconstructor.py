import itertools
import numpy as np
import math
import pickle
import glob
from time import time
import argparse
from utils.helper_fun import get_filename, read_file, find_cuts_pairs
from utils.metrics import chi2_distance
from utils.conversions import reverse_prob
from scalability.fake_reconstruct_helper_fun import fake_reconstruct
import copy
import os.path

def reconstructed_reorder(unordered,complete_path_map,smart_order):
    # print(complete_path_map)
    # print('ordering reconstructed sv')
    ordered  = [0 for sv in unordered]
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
    for idx, sv in enumerate(unordered):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uniter')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    dirname, uniter_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,device_name='fake',field='uniter_input',evaluation_method='fake')
    uniter_input = read_file(dirname+uniter_input_filename)
    dirname, plotter_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,device_name='fake',field='plotter_input',evaluation_method='fake')
    plotter_input = read_file(dirname+plotter_input_filename)
    print('-'*50,'Reconstructor','-'*50,flush=True)
    print('Existing cases:',plotter_input.keys())

    counter = len(plotter_input.keys())
    for case in uniter_input:
        if case in plotter_input:
            continue
        print('case {}'.format(case),flush=True)
        case_dict = copy.deepcopy(uniter_input[case])
        print('Cut into ',[len(x.qubits) for x in case_dict['clusters']],'clusters')

        combinations = get_combinations(case_dict['complete_path_map'])
        
        compute_begin = time()
        reconstructed_prob, scaling_factor, smart_order, total_estimated_kron_time = fake_reconstruct(complete_path_map=case_dict['complete_path_map'],
        combinations=combinations,
        full_circ=case_dict['full_circ'], cluster_circs=case_dict['clusters'],
        cluster_sim_probs=case_dict['all_cluster_prob'],run_kron=True)
        compute_time = time() - compute_begin + total_estimated_kron_time
        print('Searcher took %.3f seconds'%case_dict['searcher_time'])
        print('Quantum took %.3f seconds'%case_dict['quantum_time'])
        print('Compute took %.3f seconds'%compute_time)
        case_dict['compute_time'] = compute_time
        
        reorder_begin = time()
        reconstructed_prob = reconstructed_prob/scaling_factor
        reconstructed_prob = reconstructed_reorder(reconstructed_prob,complete_path_map=case_dict['complete_path_map'],smart_order=smart_order)
        reorder_time = time() - reorder_begin
        print('Reorder took %.3f seconds'%reorder_time)
        case_dict['reorder_time'] = reorder_time

        # reverse_begin = time()
        # norm = sum(reconstructed_prob)
        # reconstructed_prob = reconstructed_prob/norm
        # reconstructed_prob = reverse_prob(prob_l=reconstructed_prob)
        # reverse_time = time() - reverse_begin
        # print('Reverse took %.3f seconds'%reverse_time)
        # case_dict['reverse_time'] = reverse_time

        # print('reconstruction len = ', len(reconstructed_prob),'probabilities sum = ', sum(reconstructed_prob))
        assert len(reconstructed_prob) == 2**case[1]
        
        hybrid_time = case_dict['searcher_time'] + case_dict['quantum_time'] + compute_time + reorder_time
        print('QC hybrid took %.3f seconds, classical took %.3f seconds'%(hybrid_time,case_dict['std_time']))

        # [print(x,case_dict['complete_path_map'][x]) for x in case_dict['complete_path_map']]

        # pickle.dump({case:case_dict}, open('%s'%(dirname+plotter_input_filename),'ab'))
        counter += 1
        print('Reconstruction output has %d cases'%counter,flush=True)
        print('-'*100)