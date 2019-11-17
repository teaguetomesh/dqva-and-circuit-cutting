import pickle
import os
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft
import MIQCP_searcher as searcher
import cutter
from helper_fun import evaluate_circ, get_evaluator_info, find_saturated_shots
import argparse
from qiskit import IBMQ
import copy
import random

def gen_secret(num_qubit):
    num_digit = num_qubit-1
    # num = random.randint(1, 2**num_digit-1)
    num = 2**num_digit-1
    num = bin(num)[2:]
    num_with_zeros = str(num).zfill(num_digit)
    return num_with_zeros

def evaluate_full_circ(circ, total_shots, device_name):
    uniform_p = 1.0/np.power(2,len(circ.qubits))
    uniform_prob = [uniform_p for i in range(np.power(2,len(circ.qubits)))]
    print('Evaluate full circuit, %d shots'%total_shots)
    print('Evaluating fc state vector')
    sv_noiseless_fc = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)

    print('Evaluating fc qasm, %d shots'%total_shots)
    evaluator_info = {'num_shots':total_shots}
    qasm_noiseless_fc = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=evaluator_info)

    # print('Evaluating fc qasm + noise, %d shots'%total_shots)
    # evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    # fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
    # evaluator_info['num_shots'] = total_shots
    # execute_begin = time()
    # qasm_noisy_fc = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
    # print('%.3e seconds'%(time()-execute_begin))

    print('Evaluating fc hardware, %d shots'%total_shots)
    evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    fields=['device','basis_gates','coupling_map','properties','initial_layout'])
    assert np.power(2,len(circ.qubits))<evaluator_info['device'].configuration().max_experiments/3*2
    evaluator_info['num_shots'] = total_shots
    execute_begin = time()
    hw_fc = evaluate_circ(circ=circ,backend='hardware',evaluator_info=evaluator_info)
    print('Execute on hardware, %.3e seconds'%(time()-execute_begin))

    # sv_noiseless_fc = uniform_prob
    # qasm_noiseless_fc = uniform_prob
    qasm_noisy_fc = uniform_prob
    # hw_fc = uniform_prob

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
    parser.add_argument('--circuit-name', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--shots-scaling', metavar='N', type=int,help='Scaling factor for total shots')
    args = parser.parse_args()

    device_name = args.device_name
    circuit_type = args.circuit_name
    evaluator_info = get_evaluator_info(circ=None,device_name=device_name,fields=['properties','device'])
    device_size = len(evaluator_info['properties'].qubits)
    device_max_shots = evaluator_info['device'].configuration().max_shots

    # NOTE: toggle circuits to benchmark
    dimension_l = [[2,5],[3,4],[2,7],[4,4],[3,6],[4,5]]

    full_circs = {}
    all_total_shots = {}
    for cluster_max_qubit in range(args.min_qubit,args.max_qubit+1):
        for dimension in dimension_l:
            i,j = dimension
            full_circuit_size = i*j
            if full_circuit_size<=cluster_max_qubit or full_circuit_size>device_size or (cluster_max_qubit-1)*args.max_clusters<full_circuit_size:
                continue
            
            case = (cluster_max_qubit,full_circuit_size)
            print('-'*100)
            print('Case',case)

            if full_circuit_size in full_circs:
                print('Use existing full circuit')
                full_circ = full_circs[full_circuit_size]
            else:
                if circuit_type == 'supremacy':
                    full_circ = gen_supremacy(i,j,8)
                elif circuit_type == 'hwea':
                    full_circ = gen_hwea(i*j,1)
                elif circuit_type == 'bv':
                    full_circ = gen_BV(gen_secret(i*j),barriers=False)
                elif circuit_type == 'qft':
                    full_circ = gen_qft(width=i*j, barriers=False)
                else:
                    raise Exception('Illegal circuit type %s'%circuit_type)
                full_circs[full_circuit_size] = full_circ
            
            searcher_begin = time()
            hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,num_clusters=range(2,min(len(full_circ.qubits),args.max_clusters)+1),hw_max_qubit=cluster_max_qubit,evaluator_weight=1)
            searcher_time = time() - searcher_begin
            
            if m == None:
                print('Case {} not feasible'.format(case))
                print('-'*100)
                continue
            else:
                m.print_stat()
                clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
                total_shots = find_saturated_shots(clusters=clusters,complete_path_map=complete_path_map,accuracy=1e-1)
                scaled_shots = int(total_shots/args.shots_scaling)
                if scaled_shots/device_max_shots>10 or scaled_shots<1024:
                    print('Case {} requires {} jobs, {} shots'.format(case,scaled_shots/device_max_shots,scaled_shots))
                    print('-'*100)
                    continue
                all_total_shots[case] = scaled_shots
                fc_evaluations = evaluate_full_circ(full_circ,scaled_shots,device_name)
                case_dict = {'full_circ':full_circ,'fc_evaluations':fc_evaluations,'total_shots':total_shots,
                'searcher_time':searcher_time,'clusters':clusters,'complete_path_map':complete_path_map}
            try:
                evaluator_input = pickle.load(open('./benchmark_data/evaluator_input_{}_{}.p'.format(device_name,circuit_type), 'rb' ))
            except:
                evaluator_input = {}
            evaluator_input[case] = copy.deepcopy(case_dict)
            pickle.dump(evaluator_input,open('./benchmark_data/evaluator_input_{}_{}.p'.format(device_name,circuit_type),'wb'))
            print('Evaluator input cases:',evaluator_input.keys())
            print('-'*100)
    [print(case,all_total_shots[case]) for case in all_total_shots]