import pickle
import os
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
from helper_fun import evaluate_circ, get_evaluator_info, find_saturated_shots
import argparse
from qiskit import IBMQ
import copy

def evaluate_full_circ(circ, total_shots, device_name):
    print('Evaluate full circuit using %d shots'%total_shots)
    # print('Evaluating fc state vector')
    sv_noiseless_fc = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)

    # print('Evaluating fc qasm, %d shots'%total_shots)
    # evaluator_info = {'num_shots':total_shots}
    # qasm_noiseless_fc = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=evaluator_info)
    qasm_noiseless_fc = [0 for i in sv_noiseless_fc]

    # print('Evaluating fc qasm + noise, %d shots'%total_shots)
    # evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    # fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
    # evaluator_info['num_shots'] = total_shots
    # execute_begin = time()
    # qasm_noisy_fc = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
    # print('%.3e seconds'%(time()-execute_begin))
    qasm_noisy_fc = [0 for i in sv_noiseless_fc]

    # print('Evaluating fc hardware, %d shots'%total_shots)
    # evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    # fields=['device','basis_gates','coupling_map','properties','initial_layout'])
    # evaluator_info['num_shots'] = total_shots
    # if np.power(2,len(circ.qubits))<evaluator_info['device'].configuration().max_experiments/3*2:
    #     _evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,fields=['meas_filter'])
    #     evaluator_info.update(_evaluator_info)
    # execute_begin = time()
    # hw_fc = evaluate_circ(circ=circ,backend='hardware',evaluator_info=evaluator_info)
    # print('Execute on hardware, %.3e seconds'%(time()-execute_begin))
    hw_fc = [0 for i in sv_noiseless_fc]

    fc_evaluations = {'sv_noiseless':sv_noiseless_fc,
    'qasm':qasm_noiseless_fc,
    'qasm+noise':qasm_noisy_fc,
    'hw':hw_fc}

    return fc_evaluations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--min-qubit', metavar='N', type=int,help='Benchmark minimum number of HW qubits')
    parser.add_argument('--max-qubit', metavar='N', type=int,help='Benchmark maximum number of HW qubits')
    parser.add_argument('--max-clusters', metavar='N', type=int,help='max number of clusters to split into')
    parser.add_argument('--device-name', metavar='S',type=str,help='IBM device')
    args = parser.parse_args()

    device_name = args.device_name
    device_properties = get_evaluator_info(circ=None,device_name=device_name,fields=['properties'])
    device_size = len(device_properties['properties'].qubits)

    # NOTE: toggle circuits to benchmark
    dimension_l = [[2,2],[2,3],[2,4],[2,5],[3,4],[2,7],[4,4],[3,6]]

    full_circs = {}
    for cluster_max_qubit in range(args.min_qubit,args.max_qubit+1):
        for dimension in dimension_l:
            i,j = dimension
            full_circuit_size = i*j
            if full_circuit_size<=cluster_max_qubit or full_circuit_size>device_size or (cluster_max_qubit-1)*args.max_clusters<full_circuit_size:
                continue
            
            print('-'*100)
            case = (cluster_max_qubit,full_circuit_size)
            print('Case',case)

            if full_circuit_size in full_circs:
                full_circ = full_circs[full_circuit_size]['circ']
            else:
                full_circ = gen_supremacy(i,j,8)
            
            searcher_begin = time()
            hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,num_clusters=range(2,args.max_clusters+1),hw_max_qubit=cluster_max_qubit,evaluator_weight=1)
            searcher_time = time() - searcher_begin
            
            if m == None:
                print('Case {} not feasible'.format(case))
                print('-'*100)
                continue
            else:
                m.print_stat()
                clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
                total_shots = find_saturated_shots(clusters=clusters,complete_path_map=complete_path_map,accuracy=1e-1)
                fc_evaluations = evaluate_full_circ(full_circ,total_shots,device_name)
                case_dict = {'full_circ':full_circ,'fc_evaluations':fc_evaluations,
                'searcher_time':searcher_time,'clusters':clusters,'complete_path_map':complete_path_map}
            try:
                evaluator_input = pickle.load(open('./benchmark_data/evaluator_input_{}.p'.format(device_name), 'rb' ))
            except:
                evaluator_input = {}
            evaluator_input[case] = copy.deepcopy(case_dict)
            pickle.dump(evaluator_input,open('./benchmark_data/evaluator_input_{}.p'.format(device_name),'wb'))
            print('Evaluator input cases:',evaluator_input.keys())
            print('-'*100)