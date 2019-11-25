import pickle
import os
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
import MIQCP_searcher as searcher
import cutter
from helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots, reverseBits
import argparse
from qiskit import IBMQ
import copy
import math

def gen_secret(num_qubit):
    num_digit = num_qubit-1
    num = 2**num_digit-1
    num = bin(num)[2:]
    num_with_zeros = str(num).zfill(num_digit)
    return num_with_zeros

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--device-name', metavar='S',type=str,help='IBM device')
    args = parser.parse_args()
    
    dirname = './benchmark_data'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    dirname = './benchmark_data/bv'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    try:
        f = open('./benchmark_data/evaluator_input_{}_bv.p'.format(args.device_name),'rb')
        evaluator_input = pickle.load(f)
        print('Existing cases:',evaluator_input.keys())
    except:
        evaluator_input = {}

    evaluator_info = get_evaluator_info(circ=None,device_name=args.device_name,fields=['properties','device'])
    device_size = len(evaluator_info['properties'].qubits)

    # NOTE: toggle circuits to benchmark
    dimension_l = [[1,21],[1,22],[1,23],[1,24],[1,25]]
    dimension_l = [[1,7],[1,8],[1,9]]
    full_circs = {}
    cases_to_run = {}
    for dimension in dimension_l:
        i,j = dimension
        full_circuit_size = i*j
        cluster_max_qubit = math.ceil((full_circuit_size+1)/2)
        
        case = (cluster_max_qubit,full_circuit_size)
        if case in evaluator_input:
            continue
        
        print('-'*100)
        print('Case',case,flush=True)

        if full_circuit_size in full_circs:
            print('Use existing full circuit')
            full_circ = full_circs[full_circuit_size]
        else:
            full_circ = gen_BV(gen_secret(i*j),barriers=False)
        
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,num_clusters=[2],hw_max_qubit=cluster_max_qubit,evaluator_weight=0)
        searcher_time = time() - searcher_begin
        
        if m == None:
            print('Case {} not feasible'.format(case))
            print('-'*100)
            continue
        else:
            m.print_stat()
            clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
            full_circs[full_circuit_size] = full_circ
            case_dict = {'full_circ':full_circ,'searcher_time':searcher_time,
            'clusters':clusters,'complete_path_map':complete_path_map}
            cases_to_run[case] = copy.deepcopy(case_dict)
            print('%d cases to run:'%(len(cases_to_run)),cases_to_run.keys())
            print('-'*100)

for case in cases_to_run:
    print('Running case {}'.format(case))
    full_circ = cases_to_run[case]['full_circ']
    evaluator_input[case] = copy.deepcopy(cases_to_run[case])
    print('Dump evaluator_input with %d cases'%(len(evaluator_input)))
    pickle.dump(evaluator_input,open('./benchmark_data/evaluator_input_{}_bv.p'.format(args.device_name),'wb'))
    print('*'*50)
print('-'*100)