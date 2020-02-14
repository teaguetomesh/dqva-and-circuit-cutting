from utils.helper_fun import generate_circ, apply_measurement, get_filename, find_cluster_O_rho_qubit_positions, find_cuts_pairs
import utils.MIQCP_searcher as searcher
from utils.conversions import dict_to_array
from time import time
import pickle
import os
import math
import numpy as np
import argparse

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

    dirname, evaluator_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,
    device_name='fake',field='evaluator_input',evaluation_method='fake')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    circ_dict = {}
    for fc_size in range(args.min_size,args.max_size+1,2):
        circ = generate_circ(full_circ_size=fc_size,circuit_type=args.circuit_type)
        max_clusters = 3
        cluster_max_qubit = math.ceil(fc_size/1.5)
        case = (cluster_max_qubit,fc_size)
        searcher_begin = time()
        min_objective, positions, num_rho_qubits, num_O_qubits, num_d_qubits, best_num_cluster, m = searcher.find_cuts(circ=circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
        num_clusters=range(2,min(len(circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
        searcher_time = time() - searcher_begin

        if m != None:
            # m.print_stat()
            print('case {}'.format(case))
            print('Searcher time = {:.3f}, {:d} cuts'.format(searcher_time,len(positions)))
            
            std_time, std_mem = classical_resource_estimate(fc_size)
            qc_time, qc_mem = quantum_resource_estimate(num_d_qubits,num_rho_qubits,num_O_qubits)
            
            circ_dict[case] = {'full_circ':circ,'searcher_time':searcher_time,
            'num_cuts':len(positions),
            'num_d_qubits':np.array([int(x) for x in num_d_qubits]),
            'num_rho_qubits':np.array([int(x) for x in num_rho_qubits]),
            'num_O_qubits':np.array([int(x) for x in num_O_qubits]),
            'quantum_time':qc_time,'quantum_mem':qc_mem,'std_time':std_time,'std_mem':std_mem}
            
            print('qc_time = %.3f seconds, qc_mem = %f TB'%(qc_time,qc_mem))
            print('std_time = %.3f seconds, std_mem = %f TB'%(std_time,std_mem))
            print('-'*50)
    # pickle.dump(circ_dict, open(dirname+evaluator_input_filename,'wb'))