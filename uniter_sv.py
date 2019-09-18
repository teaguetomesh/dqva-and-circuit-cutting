import pickle
import glob
import itertools
import numpy as np
import math

def reverseBits(num,bitSize): 
    binary = bin(num) 
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def find_O_rho_pairs(complete_path_map,all_cluster_qubits):
    all_cluster_idx_digits = [math.ceil(math.log(x,2)) for x in all_cluster_qubits]
    print(all_cluster_idx_digits)
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        for idx, rho_qubit in enumerate(path[1:]):
            O_qubit = path[idx]
            O_qubit_reverse_idx = reverseBits(O_qubit[1],all_cluster_idx_digits[O_qubit[0]])
            rho_qubit_reverse_idx = reverseBits(rho_qubit[1],all_cluster_idx_digits[rho_qubit[0]])
            O_qubit = tuple([O_qubit[0],O_qubit[1],O_qubit_reverse_idx])
            rho_qubit = tuple([rho_qubit[0],rho_qubit[1],rho_qubit_reverse_idx])
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
    meas = list(itertools.product(range(0,2),repeat=projection.count('x')))
    for m in meas:
        m_i = 0
        full_m = [-1 for p in projection]
        for i,p in enumerate(projection):
            if p!='x':
                full_m[i] = p
            else:
                full_m[i] = m[m_i]
                m_i += 1
        res = int("".join(str(x) for x in full_m), 2)
        projected.append(cluster_sv[res])
    return projected

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
        all_cluster_projections = []
        all_cluster_inits = []
        for num_qubits in all_cluster_qubits:
            projection = ['x' for i in range(num_qubits)]
            all_cluster_projections.append(projection)
            init = [0 for i in range(num_qubits)]
            all_cluster_inits.append(init)
        print('combination:', combination)
        for idx, c in enumerate(combination):
            O_qubit, rho_qubit = O_rho_pairs[idx]
            all_cluster_projections[O_qubit[0]][O_qubit[2]] = c
            all_cluster_inits[rho_qubit[0]][rho_qubit[1]] = c
        summation_term = [1]
        for i in range(len(all_cluster_sv)):
            print('cluster %d'%i)
            init = all_cluster_inits[i]
            projection = all_cluster_projections[i]
            print('init:', init)
            print('projection:',projection)
            cluster_sv = all_cluster_sv[i][tuple(init)]
            print('original len =',len(cluster_sv))
            cluster_sv = project_sv(cluster_sv,projection)
            print('projected len =',len(cluster_sv))
            print()
            summation_term = np.kron(summation_term,cluster_sv)
        reconstructed += summation_term
        print('-'*100)
    print(reconstructed)
    print('reconstruction len = ', len(reconstructed))
    pickle.dump(reconstructed, open('%s/reconstructed.p'%dirname, 'wb' ))