import itertools
import numpy as np

def measure_basis(l):
    H = [[1,1],[1,-1]]/np.sqrt(2)
    sDagger = [[1,0],[0,-1j]]
    Id = [[1,0],[0,1]]
    if len(l)==1:
        if l[0] == 1 or l[0] == 2 or l[0] == 7 or l[0] == 8:
            return Id
        elif l[0] == 3 or l[0] == 4:
            return H
        elif l[0] == 5 or l[0] == 6:
            return np.matmul(H,sDagger)
        else:
            raise Exception('Illegal change basis')
    else:
        return np.kron(measure_basis([l[0]]), measure_basis(l[1:]))

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

def find_inits_meas(init_perm, cluster_circs, O_rho_pairs):
    cluster_inits = [[1 for qubit in cluster.qubits] for cluster in cluster_circs]
    cluster_meas = [[1 for qubit in cluster.qubits] for cluster in cluster_circs]
    for cut_idx, s in enumerate(init_perm):
        O_qubit_tuple, rho_qubit_tuple = O_rho_pairs[cut_idx]
        rho_qubit_cluster_idx, rho_qubit = rho_qubit_tuple
        cluster_idx = rho_qubit_cluster_idx
        cluster_circ = cluster_circs[cluster_idx]
        cluster_qubit_idx = cluster_circ.qubits.index(rho_qubit)
        cluster_inits[cluster_idx][cluster_qubit_idx] = 2 if s == 8 else 1 if s == 7 else s

        O_qubit_cluster_idx, O_qubit = O_qubit_tuple
        cluster_idx = O_qubit_cluster_idx
        cluster_circ = cluster_circs[cluster_idx]
        cluster_qubit_idx = cluster_circ.qubits.index(O_qubit)
        cluster_meas[cluster_idx][cluster_qubit_idx] = s

    return cluster_inits, cluster_meas

def modify_meas(cluster_meas_init,cluster_inits, cluster_meas):
    modified_cluster_meas = []
    for idx in range(len(cluster_inits)):
        init_key = cluster_inits[idx]
        meas_key = cluster_meas[idx]
        unmodified_meas = cluster_meas_init[idx][tuple(init_key)]
        if measured_in_I(meas_key):
            modified_cluster_meas.append(unmodified_meas)
        else:
            # ATTN: Qiskit measures qubits in reverse order
            change_basis_matrix = measure_basis(meas_key[::-1])
            modified_meas = np.matmul(change_basis_matrix, unmodified_meas)
            modified_cluster_meas.append(modified_meas)
    return modified_cluster_meas

def measured_in_I(meas_key):
    for basis in meas_key:
        if basis != 1 and basis != 2 and basis != 7 and basis != 8:
            return False
    return True

def multiply_sigma(modified_cluster_meas, O_rho_pairs, cluster_circs):
    sigma_cluster_meas = []
    pauli_meas_qubit_indices = [[] for x in modified_cluster_meas]
    for pair in O_rho_pairs:
        O_qubit_tuple, _ = pair
        O_qubit_cluster_idx, O_qubit = O_qubit_tuple
        cluster_qubits = cluster_circs[O_qubit_cluster_idx].qubits
        cluster_qubit_idx = len(cluster_qubits)-1-cluster_qubits.index(O_qubit)
        pauli_meas_qubit_indices[O_qubit_cluster_idx].append(cluster_qubit_idx)
    for cluster_index, cluster_meas in enumerate(modified_cluster_meas):
        num_qubits = len(cluster_circs[cluster_index].qubits)
        # print('cluster %d has %d qubits'%(cluster_index,num_qubits))
        O_qubit_positions = pauli_meas_qubit_indices[cluster_index]
        # print('O_qubit_positions :',O_qubit_positions)
        for state, prob in enumerate(cluster_meas):
            bin_state = bin(state)[2:].zfill(num_qubits)
            remainder = sum([int(bin_state[i]) for i in O_qubit_positions]) % 2
            if remainder == 1:
                # print('state %d, bin_state ='%state,bin_state, 'multiplied by -1')
                cluster_meas[state] = prob * (-1)
        # print('-'*100)
        sigma_cluster_meas.append(cluster_meas)
    return sigma_cluster_meas

def recombine(sigma_cluster_meas, complete_path_map, cluster_circs):
    total_num_qubits = len(complete_path_map)
    for final_state in range(np.power(2, total_num_qubits)):
        bin_final_state = bin(final_state)[2:].zfill(total_num_qubits)

def sampler(cluster_circs, complete_path_map, cluster_meas_init, s):
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    cluster_inits, cluster_meas = find_inits_meas(s, cluster_circs, O_rho_pairs)
    modified_cluster_meas = modify_meas(cluster_meas_init,cluster_inits, cluster_meas)
    # Convert to probabilities
    for cluster_idx, cluster_meas in enumerate(modified_cluster_meas):
        modified_cluster_meas[cluster_idx] = [np.power(abs(x),2) for x in cluster_meas]
    sigma_cluster_meas = multiply_sigma(modified_cluster_meas, O_rho_pairs, cluster_circs)
    return sigma_cluster_meas