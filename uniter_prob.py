import itertools
import numpy as np
import pickle
import glob

def read_pickle_files(dirname):
    cluster_circ_files = [f for f in glob.glob('%s/cluster_*_circ.p'%dirname)]
    all_cluster_circ = []
    for cluster_idx in range(len(cluster_circ_files)):
        cluster_circ = pickle.load(open('%s/cluster_%d_circ.p'%(dirname,cluster_idx), 'rb' ))
        all_cluster_circ.append(cluster_circ)
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))
    full_circ = pickle.load(open( '%s/full_circ.p'%dirname, 'rb' ))
    cluster_sim_prob = pickle.load(open( '%s/cluster_sim_prob.p'%dirname, 'rb' ))
    return complete_path_map, full_circ, all_cluster_circ, cluster_sim_prob

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

def update_init_meas(clusters_init_meas, O_qubit, rho_qubit, meas, init):
    _, updated_meas = clusters_init_meas[O_qubit[0]]
    updated_meas[O_qubit[1]] = meas
    clusters_init_meas[O_qubit[0]] = (clusters_init_meas[O_qubit[0]][0],updated_meas)
    updated_init, _ = clusters_init_meas[rho_qubit[0]]
    updated_init[rho_qubit[1]] = init
    clusters_init_meas[rho_qubit[0]] = (updated_init,clusters_init_meas[rho_qubit[0]][1])
    return clusters_init_meas

def find_inits_meas(cluster_circs, O_rho_pairs, s):
    # print('find initializations, measurement basis for:',s)
    clusters_init_meas = {}
    for cluster_idx, circ in enumerate(cluster_circs):
        cluster_init = ['zero' for q in circ.qubits]
        cluster_meas = ['I' for q in circ.qubits]
        clusters_init_meas[cluster_idx] = (cluster_init, cluster_meas)
    # print(clusters_init_meas)
    for pair, s_i in zip(O_rho_pairs,s):
        O_qubit, rho_qubit = pair
        if s_i == 1 or s_i == 7:
            clusters_init_meas = update_init_meas(clusters_init_meas, O_qubit, rho_qubit, 'I', 'zero')
        elif s_i == 2 or s_i == 8:
            clusters_init_meas = update_init_meas(clusters_init_meas, O_qubit, rho_qubit, 'I', 'one')
        elif s_i == 3:
            clusters_init_meas = update_init_meas(clusters_init_meas, O_qubit, rho_qubit, 'X', 'plus')
        elif s_i == 4:
            clusters_init_meas = update_init_meas(clusters_init_meas, O_qubit, rho_qubit, 'X', 'minus')
        elif s_i == 5:
            clusters_init_meas = update_init_meas(clusters_init_meas, O_qubit, rho_qubit, 'Y', 'plus_i')
        elif s_i == 6:
            clusters_init_meas = update_init_meas(clusters_init_meas, O_qubit, rho_qubit, 'Y', 'minus_i')
        else:
            raise Exception('Illegal s = %d'%s_i)
    for cluster_idx in clusters_init_meas:
        clusters_init_meas[cluster_idx] = (tuple(clusters_init_meas[cluster_idx][0]),tuple(clusters_init_meas[cluster_idx][1]))
    # print(clusters_init_meas)
    return clusters_init_meas

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
    [print(cluster_idx,correspondence_map[cluster_idx],'\n') for cluster_idx in correspondence_map]
    return correspondence_map

def multiply_sigma(cluster_prob,O_qubits_indices,s,cluster_correspondence):
    cluster_s = []
    for s_i, pair in zip(s,O_rho_pairs):
        O_qubit, _ = pair
        if O_qubit[0] == cluster_idx:
            cluster_s.append(s_i)
    # print('cluster %d O qubits:'%cluster_idx,cluster_O_qubits)
    # print('assigned s:',cluster_s)
    total_num_qubits = int(np.log2(len(cluster_prob)))
    effective_num_qubits = total_num_qubits - len(cluster_s)
    effective_states = itertools.product(range(2),repeat=effective_num_qubits)
    O_qubit_states = list(itertools.product(range(2),repeat=len(cluster_s)))
    effective_cluster_prob = []
    for state in effective_states:
        effective_state_index = int("".join(str(x) for x in state), 2)
        effective_state_prob = 0
        # print('effective state {}, index {}'.format(state,effective_state_index))
        # print('insertions = ',list(insertions))
        full_states_indices = cluster_correspondence[effective_state_index]
        for full_state_index in full_states_indices:
            
            sigma = 1
            for s_i,i in zip(cluster_s,insertion):
                # TODO: s=1,2 not considered for sigma multiplications, I don't know why
                if s_i>2 and i==1:
                    sigma *= -1
            contributing_term = sigma*cluster_prob[full_state_index]
            effective_state_prob += contributing_term
            # print('O qubit state {}, full state {}, sigma = {}, index = {}'.format(insertion,full_state,sigma,full_state_index))
            # print(contributing_term)
        # print('effective state prob = ',effective_state_prob)
        effective_cluster_prob.append(effective_state_prob)
    return effective_cluster_prob

def reconstructed_reorder(unordered,complete_path_map):
    print('ordering reconstructed sv')
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
        print('unordered %d --> ordered %d'%(idx,ordered_idx),'sv=',sv)
    return ordered

def reconstruct(complete_path_map, full_circ, cluster_circs, cluster_sim_probs):
    print('Complete path map:')
    [print(x, complete_path_map[x]) for x in complete_path_map]

    O_rho_pairs = find_cuts_pairs(complete_path_map)
    print('O rho qubits pairs:',O_rho_pairs)

    correspondence_map = effective_full_state_corresppndence(O_rho_pairs,cluster_circs)
    cluster_O_qubits = []
    for cluster_idx in range(len(cluster_circs)):
        O_qubits = []
        for pair in O_rho_pairs:
            O_qubit, _ = pair
            if O_qubit[0] == cluster_idx:
                O_qubits.append(O_qubit)
        cluster_O_qubits.append(O_qubit_idx)

    combinations = list(itertools.product(range(1,9),repeat=len(O_rho_pairs)))
    reconstructed_prob = [0 for i in range(np.power(2,len(full_circ.qubits)))]
    for s in combinations:
        print('s = ',s)
        clusters_init_meas = find_inits_meas(cluster_circs, O_rho_pairs, s)
        c_s = 1
        for s_i in s:
            if s_i == 4 or s_i == 6 or s_i == 8:
                c_s *= -1/2
            else:
                c_s *= 1/2
        t_s = [1]
        for cluster_idx, cluster_prob in enumerate(cluster_sim_probs):
            init_meas = clusters_init_meas[cluster_idx]
            cluster_prob = cluster_prob[init_meas]
            print('cluster {} selects init = {}, meas = {}'.format(cluster_idx,init_meas[0],init_meas[1]))
            cluster_prob = multiply_sigma(cluster_prob,cluster_O_qubits[cluster_idx],s,correspondence_map[cluster_idx])
            # TODO: bottleneck here
            t_s = np.kron(t_s,cluster_prob)
        reconstructed_prob += c_s*t_s
        print('-'*100)
    reconstructed_prob = reconstructed_reorder(reconstructed_prob,complete_path_map)
    print('reconstruction len = ', len(reconstructed_prob),sum(reconstructed_prob))
    return reconstructed_prob

if __name__ == '__main__':
    dirname = './data'
    complete_path_map, full_circ, cluster_circs, cluster_sim_probs = read_pickle_files(dirname)
    # reconstructed_prob = reconstruct(complete_path_map, full_circ, cluster_circs, cluster_sim_probs)
    # pickle.dump(reconstructed_prob, open('%s/reconstructed_prob.p'%dirname, 'wb'))
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    effective_full_state_corresppndence(O_rho_pairs,cluster_circs)