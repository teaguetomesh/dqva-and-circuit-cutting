import itertools
import numpy as np

def measure_basis(l):
    if len(l)==1:
        return l[0]
    else:
        return np.kron(l[0], measure_basis(l[1:]))

def find_all_s(complete_path_map):
    num_cuts = 0
    for input_qubit in complete_path_map:
        num_cuts += len(complete_path_map[input_qubit]) - 1
    return list(itertools.product(range(1,9),repeat=num_cuts))

def find_cluster_init_s(all_cluster_cut_in_qubits, s, complete_path_map, cluster_measurements):
    print('sampling for s =',s)
    cluster_init_s_map = {}
    cluster_meas_s_map = {}
    for idx, cluster_cut_in_qubits in enumerate(all_cluster_cut_in_qubits):
        cluster_init_s_map[idx] = (-1,) * len(cluster_cut_in_qubits)

    s_idx = 0
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        for cut_idx in range(len(path)-1):
            O_qubit_cluster_idx, O_qubit = path[cut_idx]
            rho_qubit_cluster_idx, rho_qubit = path[cut_idx+1]
            cluster_cut_in_qubits = all_cluster_cut_in_qubits[rho_qubit_cluster_idx]
            cluster_s_idx = cluster_cut_in_qubits.index(rho_qubit)
            cluster_s = 1 if s[s_idx]==7 else 2 if s[s_idx]==8 else s[s_idx]
            cluster_meas_s_map[(O_qubit_cluster_idx, O_qubit)] = cluster_s
            s_idx += 1
            l = list(cluster_init_s_map[rho_qubit_cluster_idx])
            l[cluster_s_idx] = cluster_s
            cluster_init_s_map[rho_qubit_cluster_idx] = tuple(l)
    
    print('cluster init s =',cluster_init_s_map)
    print('clusters meas s =',cluster_meas_s_map)
    cluster_meas_wt_init = []
    for cluster_idx in cluster_init_s_map:
        cluster_init_s = cluster_init_s_map[cluster_idx]
        cluster_meas_wt_init.append(cluster_measurements[cluster_idx][cluster_init_s])
    return cluster_meas_wt_init, cluster_meas_s_map

def modify_output_meas(cluster_meas_wt_init, cluster_meas_s_map, cluster_circs):
    cluster_meas_basis_wt_neg = [[] for x in cluster_circs]
    for cluster_idx, cluster_circ in enumerate(cluster_circs):
        cluster_meas_basis_wt_neg[cluster_idx] = [-1 for x in cluster_circ.qubits]
    for key in cluster_meas_s_map:
        cluster_idx, cluster_qubit = key
        cluster_qubit_idx = cluster_circs[cluster_idx].qubits.index(cluster_qubit)
        cluster_meas_basis_wt_neg[cluster_idx][cluster_qubit_idx] = cluster_meas_s_map[key]
        # print(key, cluster_meas_s_map[key])
        # print(cluster_qubit_idx)
        # print(cluster_circs[cluster_idx].qubits)
    cluster_meas_basis = []
    for l in cluster_meas_basis_wt_neg:
        l = [x if x!=-1 else 1 for x in l]
        cluster_meas_basis.append(l)
    
    print('cluster meas basis:', cluster_meas_basis)
    H = [[1,1],[1,-1]]/np.sqrt(2)
    sDagger = [[1,0],[0,-1j]]
    Id = [[1,0],[0,1]]

    change_basis_post_processing_l = []
    for cluster_idx, basis in enumerate(cluster_meas_basis):
        cluster_change_basis_post_processing_l = [Id for x in basis]
        for i, item in enumerate(basis):
            rev_i = len(basis) - 1 - i
            # print('cluster %d, position %d basis item ='%(cluster_idx,i), item)
            if item == 1 or item == 2:
                # print('position %d use Id'%rev_i)
                continue
            elif item == 3 or item == 4:
                # print('position %d use H'%rev_i)
                # cluster_change_basis_post_processing_l[rev_i] = 'H'
                cluster_change_basis_post_processing_l[rev_i] = H
            elif item == 5 or item == 6:
                # print('position %d use H_sDagger'%rev_i)
                # cluster_change_basis_post_processing_l[rev_i] = 'H_sDagger'
                cluster_change_basis_post_processing_l[rev_i] = np.matmul(H,sDagger)
        change_basis_post_processing_l.append(cluster_change_basis_post_processing_l)
    
    cluster_meas_wt_init_basis = []
    for cluster_idx, cluster_change_basis_post_processing_l in enumerate(change_basis_post_processing_l):
        cluster_meas = cluster_meas_wt_init[cluster_idx]
        meas_basis_matrix = measure_basis(cluster_change_basis_post_processing_l)
        cluster_meas = np.matmul(meas_basis_matrix, cluster_meas)
        cluster_meas_wt_init_basis.append(cluster_meas)

    return cluster_meas_wt_init_basis