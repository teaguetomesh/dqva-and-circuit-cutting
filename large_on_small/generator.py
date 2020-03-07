from utils.helper_fun import generate_circ, get_evaluator_info, evaluate_circ, apply_measurement, get_filename
import utils.MIP_searcher as searcher
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
        qc_mem += 2**d*4/(1024**3)
    return qc_time, qc_mem

def classical_resource_estimate(num_qubits):
    return 9*1e-6*np.exp(0.7*num_qubits), 2**num_qubits*4/(1024**3)

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
        for cluster_max_qubit in range(24,25,2):
            cluster_max_qubit = int(fc_size/3*2)
            case = (cluster_max_qubit,fc_size)
            if fc_size<=cluster_max_qubit or fc_size>24:
                print('Case {} impossible, skipped'.format(case))
                continue
            solution_dict = searcher.find_cuts(circ=circ, max_cluster_qubit=cluster_max_qubit)

            if len(solution_dict) > 0:
                m = solution_dict['model']
                d = solution_dict['num_d_qubits']
                positions = solution_dict['positions']
                searcher_time = solution_dict['searcher_time']
                num_rho_qubits = solution_dict['num_rho_qubits']
                num_O_qubits = solution_dict['num_O_qubits']
                # m.print_stat()
                print('Case {}'.format(case))
                print('MIP searcher clusters:',d)
                clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
                print('{:d} cuts --> {}, searcher time = {}'.format(K,d,searcher_time))
                
                qc_time, qc_mem = quantum_resource_estimate(d,num_rho_qubits,num_O_qubits)
                _, std_mem = classical_resource_estimate(fc_size)

                print('qc_time = %.3f seconds, qc_mem = %f GB'%(qc_time,qc_mem))
                print('std_time = %.3f seconds, std_mem = %f GB'%(std_time,std_mem))

                # if std_time>qc_time:
                case_dict = {'full_circ':circ,'clusters':clusters,'complete_path_map':complete_path_map,
                'sv':ground_truth,'hw':ground_truth,'searcher_time':solution_dict['searcher_time'],'std_time':std_time,'quantum_time':qc_time,'quantum_mem':qc_mem}
                pickle.dump({case:case_dict}, open(dirname+evaluator_input_filename,'ab'))
                print()
            else:
                print('Case {} not feasible'.format(case))
            print('-'*50)