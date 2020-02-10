import itertools
import numpy as np
import math
import pickle
import glob
from time import time
from scipy.sparse import kron, csr_matrix
from scipy.stats import wasserstein_distance
import argparse
from utils.helper_fun import get_filename, read_file
from utils.metrics import chi2_distance
from utils.conversions import reverse_prob
import copy
import os.path
from numba import jit, njit, prange

def find_cuts_pairs(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr+1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs

def find_inits_meas(cluster_circs, O_rho_pairs, s):
    # print('find initializations, measurement basis for:',s)
    clean_inits = []
    clean_meas = []
    for circ in cluster_circs:
        cluster_init = ['zero' for q in circ.qubits]
        cluster_meas = ['I' for q in circ.qubits]
        clean_inits.append(cluster_init)
        clean_meas.append(cluster_meas)
    
    clusters_init_meas = []
    cluster_meas = clean_meas
    cluster_inits = clean_inits
    for pair, s_i in zip(O_rho_pairs,s):
        O_qubit, rho_qubit = pair
        cluster_meas[O_qubit[0]][O_qubit[1]] = s_i
        cluster_inits[rho_qubit[0]][rho_qubit[1]] = s_i
    # print('inits:',cluster_inits)c
    for i,m in zip(cluster_inits,cluster_meas):
        clusters_init_meas.append((i,m))
    return clusters_init_meas

def multiply_sigma(full_cluster_prob,cluster_s,cluster_O_qubit_positions,effective_state_tranlsation):
    # print('full cluster instance prob len = ',len(full_cluster_prob))
    # print('cluster O qubits:',cluster_O_qubit_positions)
    # print('assigned s:',cluster_s)
    if len(cluster_O_qubit_positions) == 0:
        # print('no need to collapse')
        return full_cluster_prob
    
    total_num_qubits = int(np.log2(len(full_cluster_prob)))
    # effective_num_qubits = total_num_qubits - len(cluster_O_qubit_positions)
    if effective_state_tranlsation == None:
        contracted_prob = 0
        for full_state, prob in enumerate(full_cluster_prob):
            sigma = 1
            bin_full_state = bin(full_state)[2:].zfill(total_num_qubits)
            for s_i,position in zip(cluster_s,cluster_O_qubit_positions):
                O_measurement = bin_full_state[position]
                if s_i!='I' and O_measurement=='1':
                # if O_measurement=='1':
                    sigma *= -1
            # contributing_term = sigma*full_cluster_prob[full_state]
            contributing_term = sigma*prob
            contracted_prob += contributing_term
        return [contracted_prob]
    else:
        effective_cluster_prob = []
        for effective_state in effective_state_tranlsation:
            # bin_effective_state = bin(effective_state)[2:].zfill(effective_num_qubits)
            effective_state_prob = 0
            full_states = effective_state_tranlsation[effective_state]
            # print('effective state {}, binary {} = '.format(effective_state,bin_effective_state))
            for full_state in full_states:
                bin_full_state = bin(full_state)[2:].zfill(total_num_qubits)
                sigma = 1
                for s_i,position in zip(cluster_s,cluster_O_qubit_positions):
                    O_measurement = bin_full_state[position]
                    # print('s = type {} {}, O measurement = type {} {}'.format(type(s_i),s_i,type(O_measurement),O_measurement))
                    if s_i!='I' and O_measurement=='1':
                        sigma *= -1
                contributing_term = sigma*full_cluster_prob[full_state]
                effective_state_prob += contributing_term
                # print('full state {}, binary {}, {} * {} = {}'.format(full_state,bin_full_state,full_cluster_prob[full_state],sigma,contributing_term))
                # print('O qubit state {}, full state {}, sigma = {}, index = {}'.format(insertion,full_state,sigma,full_state_index))
                # print(contributing_term)
            # print(' =',effective_state_prob)
            effective_cluster_prob.append(effective_state_prob)
        # print('effective cluster inst prob len = ', len(effective_cluster_prob))
        return effective_cluster_prob

def find_cluster_O_qubit_positions(O_rho_pairs, cluster_circs):
    cluster_O_qubit_positions = {}
    for pair in O_rho_pairs:
        O_qubit, _ = pair
        cluster_idx, O_qubit_idx = O_qubit
        if cluster_idx not in cluster_O_qubit_positions:
            cluster_O_qubit_positions[cluster_idx] = [O_qubit_idx]
        else:
            cluster_O_qubit_positions[cluster_idx].append(O_qubit_idx)
    for cluster_idx in range(len(cluster_circs)):
        if cluster_idx not in cluster_O_qubit_positions:
            cluster_O_qubit_positions[cluster_idx] = []
    return cluster_O_qubit_positions

def effective_full_state_corresppndence(O_rho_pairs,cluster_circs):
    correspondence_map = {}
    for cluster_idx,circ in enumerate(cluster_circs):
        cluster_O_qubits = []
        total_num_qubits = len(circ.qubits)
        for pair in O_rho_pairs:
            O_qubit, _ = pair
            if O_qubit[0] == cluster_idx:
                cluster_O_qubits.append(O_qubit[1])
        effective_num_qubits = total_num_qubits - len(cluster_O_qubits)
        # print('cluster O qubits:',cluster_O_qubits)
        if effective_num_qubits>0:
            effective_states = itertools.product(range(2),repeat=effective_num_qubits)
            O_qubit_states = list(itertools.product(range(2),repeat=len(cluster_O_qubits)))
            cluster_correspondence = {}
            for effective_state in effective_states:
                # print('effective state:',effective_state)
                effective_state_index = int("".join(str(x) for x in effective_state), 2)
                corresponding_full_states = []
                for O_qubit_state in O_qubit_states:
                    full_state = list(effective_state)
                    for p,i in zip(cluster_O_qubits,O_qubit_state):
                        full_state.insert(p,i)
                    # print('O qubit state: {} --> full state: {}'.format(O_qubit_state,full_state))
                    full_state_index = int("".join(str(x) for x in full_state), 2)
                    corresponding_full_states.append(full_state_index)
                cluster_correspondence[effective_state_index] = corresponding_full_states
            correspondence_map[cluster_idx] = cluster_correspondence
        else:
            correspondence_map[cluster_idx] = None
    # print(correspondence_map)
    return correspondence_map

def reconstructed_reorder(unordered,complete_path_map):
    # print(complete_path_map)
    # print('ordering reconstructed sv')
    ordered = np.zeros(unordered.shape)
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
    for cluster_idx in sorted(cluster_out_qubits.keys()):
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

def calculate_cluster(cluster_idx,cluster_probs,init_meas,O_qubit_positions,effective_state_tranlsation):
    # print('O qubit positions:',O_qubit_positions)
    initilizations, measurement = init_meas
    num_effective_states = np.power(2,len(measurement)-len(O_qubit_positions))
    kronecker_term = [0 for i in range(num_effective_states)]
    # print('Cluster %d has %d effective states'%(cluster_idx,num_effective_states))
    meas = tuple([x if x!='Z' else 'I' for x in measurement])
    measurement = tuple(measurement)

    initilizations = [[x] if x == 'zero' else [x+'+',x+'-'] for x in initilizations]
    initilizations = list(itertools.product(*initilizations))
    for init in initilizations:
        # print(init,'initialized to',end=' ')
        sign = 1
        init = list(init)
        for idx,i in enumerate(init):
            if i == 'I+':
                init[idx] = 'zero'
            elif i == 'I-':
                init[idx] = 'one'
            elif i == 'X+':
                init[idx] = 'plus'
            elif i == 'X-':
                init[idx] = 'minus'
                sign *= -1
            elif i == 'Y+':
                init[idx] = 'plus_i'
            elif i == 'Y-':
                init[idx] = 'minus_i'
                sign *= -1
            elif i == 'Z+':
                init[idx] = 'zero'
            elif i == 'Z-':
                init[idx] = 'one'
                sign *= -1
            elif i == 'zero':
                continue
            else:
                raise Exception('Illegal initilization symbol :',i)
        init = tuple(init)
        # print('Cluster %d Evaluate'%cluster_idx,init,measurement)
        
        sigma_key = (init,meas,tuple([measurement[i] for i in O_qubit_positions]))
        # print('sigma key = ',sigma_key)
        effective_cluster_prob = multiply_sigma(full_cluster_prob=cluster_probs[(init,meas)],
        cluster_s=[measurement[i] for i in O_qubit_positions],
        cluster_O_qubit_positions=O_qubit_positions,
        effective_state_tranlsation=effective_state_tranlsation)
        
        if sign == 1:
            kronecker_term = [kronecker_term[i]+effective_cluster_prob[i] for i in range(len(effective_cluster_prob))]
            # print(effective_cluster_prob)
        else:
            kronecker_term = [kronecker_term[i]-effective_cluster_prob[i] for i in range(len(effective_cluster_prob))]
            # print('-1*',effective_cluster_prob)
    
    # print('length of effective cluster prob:',len(kronecker_term))
    kronecker_term = np.array(kronecker_term)
    return kronecker_term

def reconstruct(combinations, cluster_O_qubit_positions, correspondence_map, num_qubits, num_clusters, cluster_sim_probs, cluster_init_meas_combinations, scaling_factor):
    reconstruction_terms = []
    for i,s in enumerate(combinations):
        # print('s_{} = {}'.format(i,s))
        clusters_init_meas = cluster_init_meas_combinations[s]
        term = []
        for cluster_idx in range(len(cluster_sim_probs)):
            # print('Cluster {} inits meas = {}'.format(cluster_idx,clusters_init_meas[cluster_idx]))
            kronecker_term = calculate_cluster(cluster_idx=cluster_idx,
            cluster_probs=cluster_sim_probs[cluster_idx],
            init_meas=clusters_init_meas[cluster_idx],
            O_qubit_positions=cluster_O_qubit_positions[cluster_idx],
            effective_state_tranlsation=correspondence_map[cluster_idx])
            # print('cluster %d collapsed = '%cluster_idx,kronecker_term)
            term.append(kronecker_term)
        reconstruction_terms.append(term)
        # print('-'*100)
    # print()
    # print('reconstruction len = ', len(reconstructed_prob),'probabilities sum = ', sum(reconstructed_prob))
    reconstruction_terms = np.array(reconstruction_terms)
    return reconstruction_terms

def prepare_reconstruct(complete_path_map, full_circ, cluster_circs, cluster_sim_probs):
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    num_cuts = len(O_rho_pairs)
    scaling_factor = 2**num_cuts
    # print('O rho qubits pairs:',O_rho_pairs)

    basis = ['I','X','Y','Z']

    combinations = list(itertools.product(basis,repeat=len(O_rho_pairs)))
    correspondence_map = effective_full_state_corresppndence(O_rho_pairs,cluster_circs)
    # print('Effective states, full states correspondence map:')
    # [print('cluster %d' % cluster_idx,correspondence_map[cluster_idx],'\n') for cluster_idx in correspondence_map]
    cluster_O_qubit_positions = find_cluster_O_qubit_positions(O_rho_pairs, cluster_circs)

    cluster_init_meas_combinations = {}
    for s in combinations:
        clusters_init_meas = find_inits_meas(cluster_circs, O_rho_pairs, s)
        cluster_init_meas_combinations[s] = clusters_init_meas

    return combinations, cluster_O_qubit_positions, correspondence_map, scaling_factor, cluster_init_meas_combinations

def compute(reconstruction_terms,reconstructed_prob):
    for term in reconstruction_terms:
        summation_term = np.ones(1)
        for cluster_prob in term:
            summation_term = np.kron(summation_term,cluster_prob)
        reconstructed_prob += summation_term
    return reconstructed_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uniter')
    parser.add_argument('--experiment-name', metavar='S', type=str,help='which experiment to run')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device output file to reconstruct')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--shots-mode', metavar='S', type=str,help='saturated/sametotal shots mode')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend file to reconstruct')
    args = parser.parse_args()
    
    dirname, uniter_input_filename = get_filename(experiment_name=args.experiment_name,circuit_type=args.circuit_type,device_name=args.device_name,field='uniter_input',evaluation_method=args.evaluation_method)
    uniter_input = read_file(dirname+uniter_input_filename)
    dirname, plotter_input_filename = get_filename(experiment_name=args.experiment_name,circuit_type=args.circuit_type,device_name=args.device_name,field='plotter_input',evaluation_method=args.evaluation_method)
    plotter_input = read_file(dirname+plotter_input_filename)
    print('-'*50,'Reconstructor','-'*50,flush=True)
    print('Existing cases:',plotter_input.keys())

    counter = len(plotter_input.keys())
    basis = ['I','X','Y','Z']
    for case in uniter_input:
        # if case in plotter_input:
        #     continue
        if case!=(9,16):
            continue
        print('case {}'.format(case),flush=True)
        case_dict = copy.deepcopy(uniter_input[case])
        print('Cut into ',[len(x.qubits) for x in case_dict['clusters']],'clusters')

        prep_begin = time()
        combinations, cluster_O_qubit_positions, correspondence_map, scaling_factor, cluster_init_meas_combinations = prepare_reconstruct(
            complete_path_map=uniter_input[case]['complete_path_map'],
            full_circ=uniter_input[case]['full_circ'],
            cluster_circs=uniter_input[case]['clusters'],
            cluster_sim_probs=uniter_input[case]['all_cluster_prob']
        )
        reconstruction_terms = reconstruct(combinations,cluster_O_qubit_positions,correspondence_map,
        num_qubits=len(uniter_input[case]['full_circ'].qubits),num_clusters=len(uniter_input[case]['clusters']),
        cluster_sim_probs=uniter_input[case]['all_cluster_prob'],cluster_init_meas_combinations=cluster_init_meas_combinations,
        scaling_factor=scaling_factor)
        print('prep took %.2f seconds'%(time()-prep_begin))

        uniter_begin = time()
        reconstruction_length = np.prod([len(x) for x in reconstruction_terms[0]])
        reconstructed_prob = compute(reconstruction_terms,np.zeros(reconstruction_length))
        uniter_time = time()-uniter_begin
        print('Reconstruction took %.2f seconds'%uniter_time)
        
        reorder_begin = time()
        reconstructed_prob = reconstructed_reorder(reconstructed_prob,uniter_input[case]['complete_path_map'])
        norm = reconstructed_prob.sum()
        reconstructed_prob = reconstructed_prob/norm
        reconstructed_prob = reverse_prob(prob_l=reconstructed_prob)
        print('reorder took %.2f seconds'%(time()-reorder_begin))
        case_dict['reconstructor_time'] = uniter_time
        case_dict['cutting'] = reconstructed_prob
        print('qasm metric = %.3e'%chi2_distance(target=case_dict['sv'],obs=case_dict['qasm']))
        print('hw metric = %.3e'%chi2_distance(target=case_dict['sv'],obs=case_dict['hw']))
        print('cutting metric = %.3e'%(chi2_distance(target=case_dict['sv'],obs=case_dict['cutting'])))

        # pickle.dump({case:case_dict}, open('%s'%(dirname+plotter_input_filename),'ab'))
        counter += 1
        print('Reconstruction output has %d cases'%counter,flush=True)
        print('-'*100)