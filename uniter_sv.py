import pickle
import glob
import itertools
import numpy as np

def find_O_rho_pairs(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        for idx, rho_qubit in enumerate(path[1:]):
            O_qubit = path[idx]
            O_rho_pairs.append([O_qubit,rho_qubit])
    return O_rho_pairs, len(O_rho_pairs)

if __name__ == '__main__':
    dirname = './data'
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))
    [print(x, complete_path_map[x]) for x in complete_path_map]
    O_rho_pairs, num_cuts = find_O_rho_pairs(complete_path_map)
    print('%d cuts'%num_cuts)
    print(O_rho_pairs)
    combinations = list(itertools.product(range(0,2),repeat=num_cuts))

    cluster_circ_sv_files = [f for f in glob.glob('%s/cluster_*_sv.p'%dirname)]
    all_cluster_sv = []
    for cluster_idx in range(len(cluster_circ_sv_files)):
        cluster_sv = pickle.load(open('%s/cluster_%d_sv.p'%(dirname,cluster_idx), 'rb' ))
        all_cluster_sv.append(cluster_sv)
    
    for combination in combinations:
        print(combination)