import itertools
import numpy as np

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
    cluster_init = []
    cluster_meas = []
    for cluster in cluster_circs:
        init = [1 for i in cluster.qubits]
        meas = [1 for i in cluster.qubits]
        cluster_init.append(init)
        cluster_meas.append(meas)
    for idx in range(len(O_rho_pairs)):
        (O_qubit_cluster, O_qubit), (rho_qubit_cluster, rho_qubit) = O_rho_pairs[idx]
        O_rho_s = 1 if s[idx]==7 else 2 if s[idx]==8 else s[idx]
        # print(O_qubit_cluster, O_qubit, rho_qubit_cluster, rho_qubit, O_rho_s)
        O_qubit_cluster_idx = cluster_circs[O_qubit_cluster].qubits.index(O_qubit)
        rho_qubit_cluster_idx = cluster_circs[rho_qubit_cluster].qubits.index(rho_qubit)
        cluster_meas[O_qubit_cluster][O_qubit_cluster_idx] = O_rho_s
        cluster_init[rho_qubit_cluster][rho_qubit_cluster_idx] = O_rho_s

    return cluster_init, cluster_meas

def qubit_reorder(complete_path_map, cluster_circs):
    l = [[-1 for x in cluster.qubits] for cluster in cluster_circs]
    input_qubit_ctr = 0
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_cluster_idx, output_qubit = path[len(path)-1]
        output_cluster_qubits = cluster_circs[output_cluster_idx].qubits
        output_qubit_idx = len(output_cluster_qubits)-1-output_cluster_qubits.index(output_qubit)
        l[output_cluster_idx][output_qubit_idx] = input_qubit_ctr
        input_qubit_ctr += 1
    ordering = []
    for cluster_order in l:
        ordering += cluster_order
    return l

def calculate_ts(cluster_meas_init, cluster_circs, O_rho_pairs, ordering, s):
    cluster_init, cluster_meas = find_inits_meas(cluster_circs, O_rho_pairs, s)
    print(cluster_init)
    print(cluster_meas)
    cluster_sv = []
    for idx, cluster in enumerate(cluster_meas_init):
        init = tuple(cluster_init[idx])
        meas = tuple(cluster_meas[idx])
        cluster_sv.append(cluster[(init,meas)])
    t_s = list(itertools.product(*cluster_sv))
    t_s = [np.prod(x) for x in t_s[:3]]
    print(t_s)
    return t_s

def reconstruct(cluster_circs, cluster_meas_init, complete_path_map):
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    print('O_rho_pairs:')
    [print(x) for x in O_rho_pairs]
    
    ordering = qubit_reorder(complete_path_map, cluster_circs)
    print('ordering:', ordering)

    reconstructed = [0 for i in range(np.power(2,len(complete_path_map)))]
    all_s = list(itertools.product(range(1,9),repeat=len(O_rho_pairs)))
    print('%d s samples'%len(all_s))
    print('-'*100)
    for s in all_s[23435:23436]:
        print('sampling for s =', s)
        c_s = 1
        for e_s in s:
            if e_s == 4 or e_s == 6 or e_s == 8:
                c_s *= -1/2
            else:
                c_s *= 1/2
        print('c_s =', c_s)
        t_s = calculate_ts(cluster_meas_init, cluster_circs, O_rho_pairs, ordering, s)
        # t_s = [c_s * x for x in t_s]
        # reconstructed += t_s
        break
    return reconstructed