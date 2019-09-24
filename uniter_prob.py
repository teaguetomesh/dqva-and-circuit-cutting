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

if __name__ == '__main__':
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','minus','plus_i','minus_i']
    dirname = './data'
    complete_path_map, full_circ, cluster_circs, cluster_sim_probs = read_pickle_files(dirname)
    [print(x, complete_path_map[x]) for x in complete_path_map]
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    print('O rho qubits pairs:',O_rho_pairs)
    combinations = list(itertools.product(range(1,9),repeat=len(O_rho_pairs)))
    for s in combinations:
        print('s = ',s)
        clusters_init_meas = find_inits_meas(cluster_circs, O_rho_pairs, s)
        for cluster_idx, cluster_prob in enumerate(cluster_sim_probs):
            init_meas = clusters_init_meas[cluster_idx]
            cluster_prob = cluster_prob[init_meas]
            print('cluster {} selects init = {}, meas = {}'.format(cluster_idx,init_meas[0],init_meas[1]))
        print('-'*100)