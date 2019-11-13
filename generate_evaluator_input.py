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

def evaluate_full_circ(circ, device_name):
    # Evaluate full circuit
    print('Evaluating sv noiseless fc')
    sv_noiseless_fc = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)

    print('Evaluating qasm')
    evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,fields=['num_shots'])
    num_shots = evaluator_info['num_shots']
    print('evaluator fields:',evaluator_info.keys(),'Saturated = %.3e shots'%num_shots)
    qasm_noiseless_fc = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=evaluator_info)

    # print('Evaluating qasm + noise')
    # evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    # fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model','num_shots'])
    # print('evaluator fields:',evaluator_info.keys(),'Saturated = %.3e shots'%evaluator_info['num_shots'])
    # print('Execute noisy qasm simulator',end=' ')
    # execute_begin = time()
    # qasm_noisy_fc = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
    # print('%.3e seconds'%(time()-execute_begin))
    qasm_noisy_fc = [0 for i in sv_noiseless_fc]

    print('Evaluating on hardware')
    evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    fields=['device','basis_gates','coupling_map','properties','initial_layout'])
    evaluator_info['num_shots'] = num_shots
    if np.power(2,len(circ.qubits))<evaluator_info['device'].configuration().max_experiments/3*2:
        _evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,fields=['meas_filter'])
        evaluator_info.update(_evaluator_info)
    print('evaluator fields:',evaluator_info.keys(),'Saturated = %.3e shots'%evaluator_info['num_shots'])
    execute_begin = time()
    hw_fc = evaluate_circ(circ=circ,backend='hardware',evaluator_info=evaluator_info)
    print('Execute on hardware, %.3e seconds'%(time()-execute_begin))
    # hw_fc = [0 for i in sv_noiseless_fc]

    fc_evaluations = {'sv_noiseless':sv_noiseless_fc,
    'qasm':qasm_noiseless_fc,
    'qasm+noise':qasm_noisy_fc,
    'hw':hw_fc}

    return fc_evaluations, evaluator_info['num_shots']

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
    # dimension_l = [[2,2],[2,3],[3,3],[2,5],[3,4],[4,4],[4,5]]
    dimension_l = [[2,2],[2,3]]
    dirname = './benchmark_data'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    evaluator_input = {}
    full_circs = {}
    for hw_max_qubit in range(args.min_qubit,args.max_qubit+1):
        for dimension in dimension_l:
            i,j = dimension
            if i*j<=hw_max_qubit or i*j>device_size:
                continue
            
            print('-'*100)
            print('Case ',(hw_max_qubit,i*j))

            if (i*j) in full_circs:
                circ, fc_evaluations, num_shots = full_circs[(i*j)]
                # Looking for a cut
                searcher_begin = time()
                hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=circ,num_clusters=range(2,args.max_clusters+1),hw_max_qubit=hw_max_qubit,evaluator_weight=1)
                searcher_time = time() - searcher_begin
                if m == None:
                    print('Case {} not feasible'.format((hw_max_qubit,i*j)))
                    print('-'*100)
                    continue
                else:
                    m.print_stat()
                    print('Use existing full circ evaluations')
                    clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
                    # print('Complete path map:')
                    # [print(x,complete_path_map[x]) for x in complete_path_map]
                    evaluator_input[(hw_max_qubit,i*j)] = dimension,num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map
            else:
                circ = gen_supremacy(i,j,8)
                searcher_begin = time()
                hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=circ,num_clusters=range(2,args.max_clusters+1),hw_max_qubit=hw_max_qubit,evaluator_weight=1)
                searcher_time = time() - searcher_begin
                if m == None:
                    print('Case {} not feasible'.format((hw_max_qubit,i*j)))
                    print('-'*100)
                    continue
                else:
                    m.print_stat()
                    clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
                    fc_evaluations, num_shots = evaluate_full_circ(circ,device_name)
                    full_circs[(i*j)] = circ, fc_evaluations, num_shots
                    evaluator_input[(hw_max_qubit,i*j)] = dimension,num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map
            pickle.dump(evaluator_input,open('{}/evaluator_input_{}.p'.format(dirname,device_name),'wb'))
            print('Evaluator input cases:',evaluator_input.keys())
            print('-'*100)