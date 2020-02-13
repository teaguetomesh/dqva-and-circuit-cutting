from utils.helper_fun import generate_circ, get_evaluator_info, evaluate_circ, apply_measurement, get_filename
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from utils.conversions import dict_to_array
from time import time
import pickle
import os
import math
import argparse

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
        cluster_max_qubit = math.ceil(fc_size/2)
        case = (cluster_max_qubit,fc_size)
        searcher_begin = time()
        hardness, positions, ancilla, num_O_qubits, d, num_cluster, m = searcher.find_cuts(circ=circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
        num_clusters=range(2,min(len(circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
        searcher_time = time() - searcher_begin

        if m != None:
            # m.print_stat()
            print('case {}'.format(case))
            print('MIP searcher clusters:',d)
            clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
            print('{:d} cuts --> {}, searcher time = {}'.format(K,d,searcher_time))
            circ_dict[case] = {'full_circ':circ,'clusters':clusters,'complete_path_map':complete_path_map,
            'sv':ground_truth,'hw':ground_truth,'searcher_time':searcher_time,'std_time':std_time}
            print('-'*50)
    pickle.dump(circ_dict, open(dirname+evaluator_input_filename,'wb'))