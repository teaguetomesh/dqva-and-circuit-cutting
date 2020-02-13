from utils.helper_fun import generate_circ, get_evaluator_info, evaluate_circ, apply_measurement, get_filename
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from utils.conversions import dict_to_array
from time import time
import pickle
import os
import math
import argparse
import numpy as np

def quantum_resource_estimate(num_d_qubits,num_rho_qubits,num_O_qubits):
    qc_time = 0
    qc_mem = 0
    for cluster_idx in range(len(num_d_qubits)):
        d = num_d_qubits[cluster_idx]
        rho = num_rho_qubits[cluster_idx]
        O = num_O_qubits[cluster_idx]
        num_inst = 6**rho*3**O
        print('Cluster %d: %d-qubit, %d \u03C1-qubit + %d O-qubit = %d instances'%(cluster_idx,d,rho,O,num_inst))
        circuit_depth = 10
        shots = 2**d
        qc_time += num_inst*circuit_depth*500*1e-9*shots
        qc_mem += 2**d*4/1024/1024/1024/1024
    return qc_time, qc_mem

def classical_resource_estimate(num_qubits):
    return 9*1e-6*np.exp(0.7*num_qubits), 2**num_qubits*4/(1024**4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--min-size', metavar='N', type=int,help='Benchmark minimum circuit size')
    parser.add_argument('--max-size', metavar='N', type=int,help='Benchmark maximum circuit size')
    args = parser.parse_args()

    dirname, evaluator_input_filename = get_filename(experiment_name='large_on_small',circuit_type=args.circuit_type,
    device_name='ibmq_boeblingen',field='evaluator_input',evaluation_method='statevector_simulator')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    circ_dict = {}
    for fc_size in range(args.min_size,args.max_size+1,2):
        circ = generate_circ(full_circ_size=fc_size,circuit_type=args.circuit_type)
        std_begin = time()
        ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None,force_prob=True)
        ground_truth = dict_to_array(distribution_dict=ground_truth,force_prob=True)
        std_time = time() - std_begin
        max_clusters = 5
        cluster_max_qubit = math.ceil(fc_size/1.5)
        case = (cluster_max_qubit,fc_size)
        hardness, positions, num_rho_qubits, num_O_qubits, d, num_cluster, m, searcher_time = searcher.find_cuts(circ=circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
        num_clusters=range(4,min(len(circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)

        if m != None:
            # m.print_stat()
            print('case {}'.format(case))
            print('MIP searcher clusters:',d)
            clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
            print('{:d} cuts --> {}, searcher time = {}'.format(K,d,searcher_time))
            
            qc_time, qc_mem = quantum_resource_estimate(d,num_rho_qubits,num_O_qubits)
            _, std_mem = classical_resource_estimate(fc_size)

            print('qc_time = %.3f seconds, qc_mem = %f TB'%(qc_time,qc_mem))
            print('std_time = %.3f seconds, std_mem = %f TB'%(std_time,std_mem))

            circ_dict[case] = {'full_circ':circ,'clusters':clusters,'complete_path_map':complete_path_map,
            'sv':ground_truth,'hw':ground_truth,'searcher_time':searcher_time,'std_time':std_time,'quantum_time':qc_time,'quantum_mem':qc_mem}
            print('-'*50)
    pickle.dump(circ_dict, open(dirname+evaluator_input_filename,'wb'))