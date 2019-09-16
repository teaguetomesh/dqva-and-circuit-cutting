import pickle
import glob
import itertools
import numpy as np

def summation_term(combination,all_cluster_sv,complete_path_map):
    term = [1]
    print('computing combination:',combination)
    
    for cluster_idx, key_idx in enumerate(combination):
        cluster_key = list(all_cluster_sv[cluster_idx].keys())[key_idx]
        cluster_sv = all_cluster_sv[cluster_idx][cluster_key]
        term = np.kron(term,cluster_sv)
    return term

if __name__ == '__main__':
    dirname = './data'
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))

    [print(x, complete_path_map[x]) for x in complete_path_map]

    cluster_circ_sv_files = [f for f in glob.glob('%s/cluster_*_sv.p'%dirname)]
    all_cluster_sv = []
    for cluster_idx in range(len(cluster_circ_sv_files)):
        cluster_sv = pickle.load(open('%s/cluster_%d_sv.p'%(dirname,cluster_idx), 'rb' ))

        all_cluster_sv.append(cluster_sv)
    
    l = [len(cluster_sv) for cluster_sv in all_cluster_sv]
    combinations = list(itertools.product(*[range(x) for x in l]))
    term = summation_term(combinations[0],all_cluster_sv,complete_path_map)
    for combination in combinations[1:]:
        term = summation_term(combination,all_cluster_sv,complete_path_map)
        term += term
        print(len(term))