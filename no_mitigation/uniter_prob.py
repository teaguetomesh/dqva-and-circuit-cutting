import itertools
import numpy as np
import math
import pickle
import glob
from time import time
import progressbar as pb
from qiskit.quantum_info.states.measures import state_fidelity
from scipy.stats import wasserstein_distance
import argparse
from helper_fun import cross_entropy, fidelity
import copy

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
                    # TODO: s=1,2 not considered for sigma multiplications, I don't know why
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
        if effective_num_qubits>0:
            effective_states = itertools.product(range(2),repeat=effective_num_qubits)
            # print('effective states:',list(effective_states))
            O_qubit_states = list(itertools.product(range(2),repeat=len(cluster_O_qubits)))
            cluster_correspondence = {}
            for effective_state in effective_states:
                effective_state_index = int("".join(str(x) for x in effective_state), 2)
                corresponding_full_states = []
                for O_qubit_state in O_qubit_states:
                    full_state = list(effective_state)
                    for p,i in zip(cluster_O_qubits,O_qubit_state):
                        full_state.insert(p,i)
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

def calculate_cluster(cluster_idx,cluster_probs,init_meas,O_qubit_positions,effective_state_tranlsation,collapsed_cluster_prob):
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
        if sigma_key not in collapsed_cluster_prob[cluster_idx]:
            effective_cluster_prob = multiply_sigma(full_cluster_prob=cluster_probs[(init,meas)],
            cluster_s=[measurement[i] for i in O_qubit_positions],
            cluster_O_qubit_positions=O_qubit_positions,
            effective_state_tranlsation=effective_state_tranlsation)
            collapsed_cluster_prob[cluster_idx][sigma_key] = effective_cluster_prob
        else:
            effective_cluster_prob = collapsed_cluster_prob[cluster_idx][sigma_key]
        
        if sign == 1:
            kronecker_term = [kronecker_term[i]+effective_cluster_prob[i] for i in range(len(effective_cluster_prob))]
            # print(effective_cluster_prob)
        else:
            kronecker_term = [kronecker_term[i]-effective_cluster_prob[i] for i in range(len(effective_cluster_prob))]
            # print('-1*',effective_cluster_prob)
    
    # print('length of effective cluster prob:',len(kronecker_term))
    
    return kronecker_term, collapsed_cluster_prob

# TODO: optimize this
def reconstruct(complete_path_map, full_circ, cluster_circs, cluster_sim_probs):
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    num_cuts = len(O_rho_pairs)
    scaling_factor = np.power(2,num_cuts)
    # print('O rho qubits pairs:',O_rho_pairs)

    basis = ['I','X','Y','Z']

    combinations = list(itertools.product(basis,repeat=len(O_rho_pairs)))
    reconstructed_prob = [0 for i in range(np.power(2,len(full_circ.qubits)))]
    correspondence_map = effective_full_state_corresppndence(O_rho_pairs,cluster_circs)
    # print('Effective states, full states correspondence map:')
    # [print('cluster %d' % cluster_idx,correspondence_map[cluster_idx],'\n') for cluster_idx in correspondence_map]
    cluster_O_qubit_positions = find_cluster_O_qubit_positions(O_rho_pairs, cluster_circs)

    # bar = pb.ProgressBar(max_value=len(combinations))
    collapsed_cluster_prob = [{} for c in cluster_circs]
    for i,s in enumerate(combinations):
        # print('s_{} = {}'.format(i,s))
        clusters_init_meas = find_inits_meas(cluster_circs, O_rho_pairs, s)
        summation_term = [1]
        for cluster_idx in range(len(cluster_circs)):
            # print('Cluster {} inits meas = {}'.format(cluster_idx,clusters_init_meas[cluster_idx]))
            kronecker_term, collapsed_cluster_prob = calculate_cluster(cluster_idx=cluster_idx,
            cluster_probs=cluster_sim_probs[cluster_idx],
            init_meas=clusters_init_meas[cluster_idx],
            O_qubit_positions=cluster_O_qubit_positions[cluster_idx],
            effective_state_tranlsation=correspondence_map[cluster_idx],
            collapsed_cluster_prob=collapsed_cluster_prob)
            # print('cluster %d collapsed = '%cluster_idx,kronecker_term)
            summation_term = np.kron(summation_term,kronecker_term)
        reconstructed_prob += summation_term
        # bar.update(i)
        # print('-'*100)
    # print()
    reconstructed_prob = [x/scaling_factor for x in reconstructed_prob]
    reconstructed_prob = reconstructed_reorder(reconstructed_prob,complete_path_map)
    # print('reconstruction len = ', len(reconstructed_prob),'probabilities sum = ', sum(reconstructed_prob))
    return reconstructed_prob

def get_filename(device_name,circuit_type,shots_mode,evaluation_method):
    dirname = './benchmark_data/{}/'.format(circuit_type)
    if evaluation_method == 'statevector_simulator':
        filename = 'classical_uniter_input_{}_{}.p'.format(device_name,circuit_type)
    elif evaluation_method == 'noisy_qasm_simulator':
        filename = 'quantum_uniter_input_{}_{}_{}.p'.format(device_name,circuit_type,shots_mode)
    elif evaluation_method == 'hardware':
        filename = 'hardware_uniter_input_{}_{}_{}.p'.format(device_name,circuit_type,shots_mode)
    else:
        raise Exception('Illegal evaluation method :',evaluation_method)
    return dirname+filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uniter')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device output file to reconstruct')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--shots-mode', metavar='S', type=str,help='saturated/sametotal shots mode')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend file to reconstruct')
    args = parser.parse_args()

    input_file = get_filename(device_name=args.device_name,circuit_type=args.circuit_type,shots_mode=args.shots_mode,evaluation_method=args.evaluation_method)
    filename = input_file.replace('uniter_input','plotter_input')
    print('-'*50,'Reconstructing %s'%input_file,'-'*50)

    try:
        f = open(filename,'rb')
        uniter_output = {}
        while 1:
            try:
                uniter_output.update(pickle.load(f))
            except (EOFError):
                break
        f.close()
    except:
        uniter_output = {}

    evaluator_output = pickle.load(open(input_file, 'rb' ) )
    for case in evaluator_output:
        print('case {}'.format(case))
        if case in uniter_output:
            continue
        case_dict = copy.deepcopy(evaluator_output[case])
        evaluations = evaluator_output[case]['fc_evaluations']
        
        uniter_begin = time()
        reconstructed_prob = reconstruct(complete_path_map=evaluator_output[case]['complete_path_map'],
        full_circ=evaluator_output[case]['full_circ'], cluster_circs=evaluator_output[case]['clusters'],
        cluster_sim_probs=evaluator_output[case]['all_cluster_prob'])
        uniter_time = time()-uniter_begin
        evaluations['cutting'] = reconstructed_prob
        
        ground_truth_ce = cross_entropy(target=evaluations['sv_noiseless'],obs=evaluations['sv_noiseless'])
        fc_ce = cross_entropy(target=evaluations['sv_noiseless'],obs=evaluations['qasm+noise'])
        cutting_ce = cross_entropy(target=evaluations['sv_noiseless'],obs=evaluations['cutting'])
        ce_percent_change = 100*(fc_ce-cutting_ce)/(fc_ce-ground_truth_ce)
        distance = wasserstein_distance(evaluations['sv_noiseless'],evaluations['cutting'])

        ground_truth_fid = fidelity(target=evaluations['sv_noiseless'],obs=evaluations['sv_noiseless'])
        fc_fid = fidelity(target=evaluations['sv_noiseless'],obs=evaluations['qasm+noise'])
        cutting_fid = fidelity(target=evaluations['sv_noiseless'],obs=evaluations['cutting'])
        fid_percent_change = 100*(cutting_fid-fc_fid)/fc_fid
        print('reconstruction distance = {:.3f}, ce percent reduction = {:.3f}, fidelity improvement = {:.3f}, time = {:.3e}'.format(distance,ce_percent_change,fid_percent_change,uniter_time),flush=True)
        assert fc_ce+1e-5>=ground_truth_ce and cutting_ce+1e-5>=ground_truth_ce and (ce_percent_change<=100+1e-5 or math.isnan(ce_percent_change))

        case_dict['evaluations'] = copy.deepcopy(evaluations)
        case_dict['ce_percent_reduction'] = copy.deepcopy(ce_percent_change)
        case_dict['fid_percent_improvement'] = copy.deepcopy(fid_percent_change)
        case_dict['uniter_time'] = copy.deepcopy(uniter_time)
        uniter_output.update({case:case_dict})
        pickle.dump({case:case_dict}, open('%s'%filename,'ab'))
        print('Reconstruction output has %d cases'%(len(uniter_output)),flush=True)
        print('-'*100)