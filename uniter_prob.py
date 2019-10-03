import itertools
import numpy as np
import pickle
import glob
from time import time
from scipy.stats import wasserstein_distance
import progressbar as pb

def read_pickle_files(dirname):
    cluster_circ_files = [f for f in glob.glob('%s/cluster_*_circ.p'%dirname)]
    all_cluster_circ = []
    for cluster_idx in range(len(cluster_circ_files)):
        cluster_circ = pickle.load(open('%s/cluster_%d_circ.p'%(dirname,cluster_idx), 'rb' ))
        all_cluster_circ.append(cluster_circ)
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))
    full_circ = pickle.load(open( '%s/full_circ.p'%dirname, 'rb' ))
    cluster_sim_prob = pickle.load(open( '%s/cluster_sim_prob.p'%dirname, 'rb' ))
    full_circ_sim_prob = pickle.load(open( '%s/full_circ_sim_prob.p'%dirname, 'rb' ))
    return complete_path_map, full_circ, all_cluster_circ, cluster_sim_prob,full_circ_sim_prob

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
    # print('cluster O qubits:',cluster_O_qubit_positions)
    # print('assigned s:',cluster_s)
    if len(cluster_O_qubit_positions) == 0:
        return full_cluster_prob
    total_num_qubits = int(np.log2(len(full_cluster_prob)))
    effective_cluster_prob = []
    for effective_state in effective_state_tranlsation:
        # effective_state_index = int("".join(str(x) for x in state), 2)
        effective_state_prob = 0
        full_states = effective_state_tranlsation[effective_state]
        # print('effective state {}, index {}'.format(state,effective_state_index))
        for full_state in full_states:
            bin_full_state = bin(full_state)[2:].zfill(total_num_qubits)
            sigma = 1
            for s_i,position in zip(cluster_s,cluster_O_qubit_positions):
                O_measurement = bin_full_state[position]
                # print('s = type {} {}, O measurement = type {} {}'.format(type(s_i),s_i,type(O_measurement),O_measurement))
                # TODO: s=1,2 not considered for sigma multiplications, I don't know why
                if s_i!='I' and O_measurement=='1':
                    sigma *= -1
            # print('full state {}, binary {}, sigma = {}'.format(full_state,bin_full_state,sigma))
            contributing_term = sigma*full_cluster_prob[full_state]
            effective_state_prob += contributing_term
            # print('O qubit state {}, full state {}, sigma = {}, index = {}'.format(insertion,full_state,sigma,full_state_index))
            # print(contributing_term)
        # print('effective state prob = ',effective_state_prob)
        effective_cluster_prob.append(effective_state_prob)
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
    # print(correspondence_map)
    return correspondence_map

def reconstructed_reorder(unordered,complete_path_map):
    # print('ordering reconstructed sv')
    ordered  = [0 for sv in unordered]
    cluster_out_qubits = {}
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        # print('output qubit = ', output_qubit)
        if output_qubit[0] in cluster_out_qubits:
            cluster_out_qubits[output_qubit[0]].append((output_qubit[1],input_qubit[1]))
        else:
            cluster_out_qubits[output_qubit[0]] = [(output_qubit[1],input_qubit[1])]
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
        # print(init,measurement)
        
        sigma_key = (init,meas,tuple([measurement[i] for i in O_qubit_positions]))
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
        else:
            kronecker_term = [kronecker_term[i]-effective_cluster_prob[i] for i in range(len(effective_cluster_prob))]
    
    # print('length of effective cluster prob:',len(kronecker_term))
    
    return kronecker_term, collapsed_cluster_prob

def reconstruct(complete_path_map, full_circ, cluster_circs, cluster_sim_probs):
    print('Reconstructing')

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

    bar = pb.ProgressBar(max_value=len(combinations))
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
            summation_term = np.kron(summation_term,kronecker_term)
        reconstructed_prob += summation_term
        bar.update(i)
        # print('-'*100)
    print()
    reconstructed_prob = [x/scaling_factor for x in reconstructed_prob]
    reconstructed_prob = reconstructed_reorder(reconstructed_prob,complete_path_map)
    print('reconstruction len = ', len(reconstructed_prob),'probabilities sum = ', sum(reconstructed_prob))
    return reconstructed_prob

if __name__ == '__main__':
    begin = time()
    dirname = './data'
    complete_path_map, full_circ, cluster_circs, cluster_sim_probs,full_circ_sim_prob = read_pickle_files(dirname)
    reconstructed_prob = reconstruct(complete_path_map, full_circ, cluster_circs, cluster_sim_probs)
    # pickle.dump(reconstructed_prob, open('%s/reconstructed_prob.p'%dirname, 'wb'))
    print('Python time elapsed = %f seconds'%(time()-begin))
    distance = wasserstein_distance(full_circ_sim_prob,reconstructed_prob)
    print('probability reconstruction distance = ',distance)
    print('first element comparison: full_circ = ',full_circ_sim_prob[0],'reconstructed = ',reconstructed_prob[0])