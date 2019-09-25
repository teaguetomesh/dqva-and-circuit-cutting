import pickle
import glob
import itertools
import numpy as np
import math

def find_O_rho_pairs(complete_path_map,all_cluster_qubits):
    all_cluster_idx_digits = [math.ceil(math.log(x,2)) for x in all_cluster_qubits]
    print(all_cluster_idx_digits)
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        for idx, rho_qubit in enumerate(path[1:]):
            O_qubit = path[idx]
            # O_qubit_reverse_idx = all_cluster_qubits[O_qubit[0]]-1-O_qubit[1]
            O_qubit = tuple([O_qubit[0],O_qubit[1]])
            rho_qubit = tuple([rho_qubit[0],rho_qubit[1]])
            O_rho_pairs.append([O_qubit,rho_qubit])
    return O_rho_pairs

def find_combinations(O_rho_pairs):
    num_cuts = len(O_rho_pairs)
    print('%d cuts'%num_cuts)
    print('cut qubit pairs:\n',O_rho_pairs)
    combinations = list(itertools.product(range(0,2),repeat=num_cuts))
    return combinations

def read_sv_files(dirname):
    cluster_circ_sv_files = [f for f in glob.glob('%s/cluster_*_sv.p'%dirname)]
    all_cluster_sv = []
    all_cluster_qubits = []
    for cluster_idx in range(len(cluster_circ_sv_files)):
        cluster_sv = pickle.load(open('%s/cluster_%d_sv.p'%(dirname,cluster_idx), 'rb' ))
        all_cluster_sv.append(cluster_sv)
        cluster_circ = pickle.load(open('%s/cluster_%d_circ.p'%(dirname,cluster_idx), 'rb' ))
        all_cluster_qubits.append(len(cluster_circ.qubits))
    return all_cluster_sv, all_cluster_qubits

def project_sv(cluster_sv,projection):
    projected = []
    for i, sv in enumerate(cluster_sv):
        bin_i = bin(i)[2:].zfill(len(projection))
        pattern_match = True
        for b, p in zip(bin_i, projection):
            b = int(b)
            if b!=p and p!='x':
                pattern_match = False
                break
        if pattern_match:
            projected.append(sv)
    return projected

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
    print(cluster_out_qubits)
    for cluster_idx in cluster_out_qubits:
        cluster_out_qubits[cluster_idx].sort()
        cluster_out_qubits[cluster_idx] = [x[1] for x in cluster_out_qubits[cluster_idx]]
    print(cluster_out_qubits)
    unordered_qubit_idx = []
    for cluster_idx in sorted(cluster_out_qubits.keys()):
        unordered_qubit_idx += cluster_out_qubits[cluster_idx]
    print(unordered_qubit_idx)
    for idx, sv in enumerate(unordered):
        bin_idx = bin(idx)[2:].zfill(len(unordered_qubit_idx))
        print('sv bin_idx=',bin_idx)
        ordered_idx = [0 for i in unordered_qubit_idx]
        for jdx, i in enumerate(bin_idx):
            ordered_idx[unordered_qubit_idx[jdx]] = i
        print(ordered_idx)
        ordered_idx = int("".join(str(x) for x in ordered_idx), 2)
        ordered[ordered_idx] = sv
        print('unordered %d --> ordered %d'%(idx,ordered_idx),'sv=',sv)
    return ordered

if __name__ == '__main__':
    dirname = './data'
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))
    full_circ = pickle.load(open( '%s/full_circ.p'%dirname, 'rb' ))
    [print(x, complete_path_map[x]) for x in complete_path_map]
    all_cluster_sv, all_cluster_qubits = read_sv_files(dirname)
    O_rho_pairs= find_O_rho_pairs(complete_path_map,all_cluster_qubits)
    combinations = find_combinations(O_rho_pairs)
    
    print('*'*100)
    print('start reconstruction')
    reconstructed = [0 for i in range(np.power(2,len(full_circ.qubits)))]
    for combination in combinations:
        print('combination:', combination)
        # Initialize initializations and projections
        all_cluster_projections = []
        all_cluster_inits = []
        for num_qubits in all_cluster_qubits:
            projection = ['x' for i in range(num_qubits)]
            all_cluster_projections.append(projection)
            init = [0 for i in range(num_qubits)]
            all_cluster_inits.append(init)
        for idx, c in enumerate(combination):
            O_qubit, rho_qubit = O_rho_pairs[idx]
            all_cluster_projections[O_qubit[0]][O_qubit[1]] = c
            all_cluster_inits[rho_qubit[0]][rho_qubit[1]] = c
        summation_term = [1]
        for cluster_idx in range(len(all_cluster_sv)):
            print('cluster %d'%cluster_idx)
            init = all_cluster_inits[cluster_idx]
            projection = all_cluster_projections[cluster_idx]
            print('init:', init)
            print('projection:',projection)
            cluster_sv = all_cluster_sv[cluster_idx][tuple(init)]
            print('original len =',len(cluster_sv))
            cluster_sv = project_sv(cluster_sv,projection)
            print('projected len =',len(cluster_sv))
            print()
            summation_term = np.kron(summation_term,cluster_sv)
        # print('summation term =', summation_term)
        reconstructed += summation_term
        print('-'*100)
    # TODO: reordering required here
    # print('unordered reconstruction:\n',reconstructed)
    reconstructed = reconstructed_reorder(reconstructed,complete_path_map)
    # print(reconstructed)
    print('reconstruction len = ', len(reconstructed))
    pickle.dump(reconstructed, open('%s/reconstructed_sv.p'%dirname, 'wb'))