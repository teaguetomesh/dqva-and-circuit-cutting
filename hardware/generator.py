import pickle
import os
from time import time
import numpy as np
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from utils.helper_fun import get_evaluator_info, get_circ_saturated_shots, get_filename, read_file, generate_circ,evaluate_circ
from utils.submission import Scheduler
import argparse
import math
import copy

def case_feasible(full_circ,cluster_max_qubit,max_clusters):
    searcher_begin = time()
    hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
    num_clusters=range(2,min(len(full_circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
    searcher_time = time() - searcher_begin
    if m == None:
        return None
    else:
        m.print_stat()
        clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
        return {'full_circ':full_circ,'searcher_time':searcher_time,'clusters':clusters,'complete_path_map':complete_path_map}

def fc_size_in_dict(full_circ_size,dictionary):
    for case in dictionary:
        if full_circ_size == case[1]:
            return True, dictionary[case]['full_circ']
    return False, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--min-qubit', metavar='N', type=int,help='Benchmark minimum number of HW qubits')
    parser.add_argument('--max-qubit', metavar='N', type=int,help='Benchmark maximum number of HW qubits')
    parser.add_argument('--max-clusters', metavar='N', type=int,help='max number of clusters to split into')
    parser.add_argument('--device-name', metavar='S',type=str,help='IBM device')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    dirname, evaluator_input_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='evaluator_input',evaluation_method=None,shots_mode=None)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print('-'*50,'Generator','-'*50,flush=True)
    evaluator_input = read_file(dirname+evaluator_input_filename)
    print('Existing cases:',evaluator_input.keys())

    evaluator_info = get_evaluator_info(circ=None,device_name=args.device_name,fields=['properties','device'])
    device_size = len(evaluator_info['properties'].qubits)

    full_circuit_sizes = np.arange(3,11)
    cases_to_run = {}
    circ_dict = {}
    for full_circ_size in full_circuit_sizes:
        if full_circ_size in circ_dict:
            full_circ = circ_dict[full_circ_size]['circ']
        elif fc_size_in_dict(full_circ_size=full_circ_size,dictionary=evaluator_input)[0]==True:
            full_circ = fc_size_in_dict(full_circ_size=full_circ_size,dictionary=evaluator_input)[1]
        else:
            full_circ = generate_circ(full_circ_size=full_circ_size,circuit_type=args.circuit_type)
        for cluster_max_qubit in range(args.min_qubit,args.max_qubit+1):
            case = (cluster_max_qubit,full_circ_size)
            if full_circ_size<=cluster_max_qubit or full_circ_size>device_size or (cluster_max_qubit-1)*args.max_clusters<full_circ_size:
                print('Case {} impossible, skipped'.format(case))
            elif case in evaluator_input:
                print('Case {} already exists'.format(case))
            else:
                feasibility = case_feasible(full_circ=full_circ,cluster_max_qubit=cluster_max_qubit,max_clusters=args.max_clusters)
                if feasibility == None:
                    print('Case {} NOT feasible'.format(case))
                else:
                    print('Adding case {} to run'.format(case))
                    cases_to_run[case] = copy.deepcopy(feasibility)

                    if full_circ_size not in circ_dict:
                        print('Adding %d qubit full circuit to run'%full_circ_size)
                        saturated_shots, ground_truths, saturated_probs = get_circ_saturated_shots(circs=[full_circ],device_name=args.device_name)
                        circ_dict[full_circ_size] = copy.deepcopy({'circ':full_circ,'shots':saturated_shots[0],
                        'sv':ground_truths[0],'qasm':saturated_probs[0]})
                    else:
                        print('Use currently running %d qubit full circuit'%full_circ_size)
            print('-'*100)
    print('{:d} cases, {:d} full circuits to run : {}'.format(len(cases_to_run),len(circ_dict),cases_to_run.keys()))
    
    scheduler = Scheduler(circ_dict=circ_dict,device_name=args.device_name)
    schedule = scheduler.get_schedule()
    jobs = scheduler.submit_schedule(schedule=schedule)
    scheduler.retrieve(schedule=schedule,jobs=jobs)
    circ_dict = scheduler.circ_dict

    for case in cases_to_run:
        uniform_prob = [1/2**case[1] for i in range(2**case[1])]
        case_dict = {'full_circ':circ_dict[case[1]]['circ'],'fc_shots':circ_dict[case[1]]['shots'],
        'sv':circ_dict[case[1]]['sv'],'qasm':circ_dict[case[1]]['qasm'],'qasm+noise':uniform_prob,'hw':circ_dict[case[1]]['hw'],
        'searcher_time':cases_to_run[case]['searcher_time'],'clusters':cases_to_run[case]['clusters'],'complete_path_map':cases_to_run[case]['complete_path_map']}
        # print('Case {}: {:d} qubit full circuit has {:d} clusters, searcher time = {:.3e}'.format(case,len(case_dict['full_circ'].qubits),len(case_dict['clusters']),case_dict['searcher_time']))
        assert case[1] == len(case_dict['full_circ'].qubits)
        for key in ['sv','qasm','qasm+noise','hw']:
            assert len(case_dict[key])==2**case[1] and abs(sum(case_dict[key])-1)<1e-10
        pickle.dump({case:case_dict},open(dirname+evaluator_input_filename,'ab'))