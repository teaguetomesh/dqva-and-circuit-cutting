from utils.helper_fun import generate_circ, apply_measurement, get_filename, find_cluster_O_rho_qubit_positions, find_cuts_pairs
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from utils.conversions import dict_to_array
from time import time
import pickle
import os
import math
import numpy as np

def cluster_runtime_estimate(complete_path_map,clusters):
    O_rho_pairs = find_cuts_pairs(complete_path_map=complete_path_map)
    cluster_O_qubit_positions, cluster_rho_qubit_positions = find_cluster_O_rho_qubit_positions(O_rho_pairs=O_rho_pairs,cluster_circs=clusters)
    total_QC_time = 0
    for cluster_idx in cluster_O_qubit_positions:
        cluster_O_qubits = cluster_O_qubit_positions[cluster_idx]
        cluster_rho_qubits = cluster_rho_qubit_positions[cluster_idx]
        num_qubits = len(clusters[cluster_idx].qubits)
        depth = clusters[cluster_idx].depth()
        shots = 2**num_qubits
        single_time = 500*1e-9*depth*shots
        reps = 6**len(cluster_rho_qubits)*3**len(cluster_O_qubits)
        estimated_time = reps * single_time
        total_QC_time += estimated_time
        # print('%d O qubits, %d rho qubits, %d qubit circuit depth %d'%(len(cluster_O_qubits),len(cluster_rho_qubits),num_qubits,depth))
        # print('estimated_time = %.3f * %d = %.5f seconds'%(single_time,reps,estimated_time))
    # print('Total QC runtime = %.5f'%total_QC_time)
    return total_QC_time

def classical_time(num_qubits):
    return 9*1e-6*np.exp(0.7*num_qubits)

if __name__ == '__main__':
    dirname, evaluator_input_filename = get_filename(experiment_name='large_on_small',circuit_type='supremacy',
    device_name='fake',field='evaluator_input',evaluation_method='statevector_simulator')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    circ_dict = {}
    for fc_size in range(16,17,2):
        circ = generate_circ(full_circ_size=fc_size,circuit_type='supremacy')
        max_clusters = 3
        cluster_max_qubit = math.ceil(fc_size/1.5)
        case = (cluster_max_qubit,fc_size)
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
        num_clusters=range(2,min(len(circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
        searcher_time = time() - searcher_begin

        std_time = classical_time(fc_size)

        if m != None:
            # m.print_stat()
            print('case {}'.format(case))
            print('MIP searcher clusters:',d)
            clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
            print('{:d} cuts --> {}, searcher time = {}'.format(K,d,searcher_time))
            total_QC_time = cluster_runtime_estimate(complete_path_map,clusters)
            circ_dict[case] = {'full_circ':circ,'clusters':clusters,'complete_path_map':complete_path_map,
            'searcher_time':searcher_time,'quantum_time':total_QC_time,'std_time':std_time}
            print('-'*50)
    pickle.dump(circ_dict, open(dirname+evaluator_input_filename,'wb'))