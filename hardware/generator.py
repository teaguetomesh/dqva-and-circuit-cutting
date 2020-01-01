import pickle
import os
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from utils.helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots, readout_mitigation, reverseBits, get_filename, read_file, factor_int, generate_circ
from utils.submission import Scheduler
import argparse
from qiskit import IBMQ
import math
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
import copy

def accumulate_jobs(jobs,meas_filter):
    hw_counts = {}
    if meas_filter != None:
        meas_filter_job, state_labels, qubit_list = meas_filter
        print('Meas filter job id {}'.format(meas_filter_job.job_id()),flush=True)
        cal_results = meas_filter_job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
        meas_filter = meas_fitter.filter
    for item in jobs:
        job = item['job']
        circ = item['circ']
        mapped_circuit_l = item['mapped_circuit_l']
        evaluator_info = item['evaluator_info']
        print('job_id : {}'.format(job.job_id()))
        hw_result = job.result()
        if meas_filter != None:
            mitigation_begin = time()
            hw_result = meas_filter.apply(hw_result)
            print('Mitigation for %d * %d qubit circuit took %.3e seconds'%(len(hw_result.results),len(circ.qubits),time()-mitigation_begin))
        for idx in range(len(mapped_circuit_l)):
            experiment_hw_counts = hw_result.get_counts(idx)
            for state in experiment_hw_counts:
                if state not in hw_counts:
                    hw_counts[state] = experiment_hw_counts[state]
                else:
                    hw_counts[state] += experiment_hw_counts[state]
    # Note that after mitigation, total number of shots may not be an integer anymore. Checking its sum does not make sense
    hw_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
    for state in hw_counts:
        reversed_state = reverseBits(int(state,2),len(circ.qubits))
        hw_prob[reversed_state] = hw_counts[state]/evaluator_info['num_shots']
    return hw_prob

def evaluate_full_circ(circ, total_shots, device_name, fields):
    uniform_p = 1.0/np.power(2,len(circ.qubits))
    uniform_prob = [uniform_p for i in range(np.power(2,len(circ.qubits)))]
    fc_evaluations = {}

    if 'sv' in fields:
        print('Evaluating fc state vector')
        sv_noiseless_fc = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)
    else:
        sv_noiseless_fc = uniform_prob
    
    if 'qasm' in fields:
        print('Evaluating fc qasm, %d shots'%total_shots)
        qasm_evaluator_info = {'num_shots':total_shots}
        qasm_noiseless_fc = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info)
    else:
        qasm_noiseless_fc = uniform_prob

    if 'qasm+noise' in fields:
        print('Evaluating fc qasm + noise, %d shots'%total_shots)
        qasm_noise_evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
        fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
        qasm_noise_evaluator_info['num_shots'] = total_shots
        execute_begin = time()
        qasm_noisy_fc = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=qasm_noise_evaluator_info)
        print('%.3e seconds'%(time()-execute_begin))
    else:
        qasm_noisy_fc = uniform_prob

    if 'hw' in fields:
        print('Evaluating fc hardware, %d shots'%total_shots)
        hw_evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
        fields=['device','basis_gates','coupling_map','properties','initial_layout'])
        hw_evaluator_info['num_shots'] = total_shots
        hw_jobs = evaluate_circ(circ=circ,backend='hardware',evaluator_info=hw_evaluator_info)
        if np.power(2,len(circ.qubits))<hw_evaluator_info['device'].configuration().max_experiments/3*2:
            meas_filter_job, state_labels, qubit_list = readout_mitigation(device=hw_evaluator_info['device'],initial_layout=hw_evaluator_info['initial_layout'])
            fc_evaluations['meas_filter'] = (meas_filter_job, state_labels, qubit_list)
    else:
        hw_jobs = uniform_prob

    fc_evaluations.update({'sv':sv_noiseless_fc,
    'qasm':qasm_noiseless_fc,
    'qasm+noise':qasm_noisy_fc,
    'hw':hw_jobs})

    return fc_evaluations

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
        return {'searcher_time':searcher_time,'clusters':clusters,'complete_path_map':complete_path_map}

def fc_size_in_dict(full_circ_size,dictionary):
    for case in dictionary:
        if full_circ_size == case[1]:
            return dictionary[case]['full_circ']
    return None

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

    full_circuit_sizes = np.arange(3,8)
    cases_to_run = []
    full_circ_to_run = {}
    for full_circ_size in full_circuit_sizes:
        if full_circ_size in full_circ_to_run:
            full_circ = full_circ_to_run[full_circ_size]['circ']
        elif fc_size_in_dict(full_circ_size=full_circ_size,dictionary=evaluator_input)!=None:
            full_circ = fc_size_in_dict(full_circ_size=full_circ_size,dictionary=evaluator_input)
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
                    cases_to_run.append(case)
                    searcher_time = feasibility['searcher_time']
                    clusters = feasibility['clusters']
                    complete_path_map = feasibility['complete_path_map']

                    if full_circ_size not in full_circ_to_run:
                        print('Adding %d qubit full circuit to run'%full_circ_size)
                        saturated_shots, saturated_probs, ground_truths = get_circ_saturated_shots(circs=[full_circ],device_name=args.device_name)
                        full_circ_to_run[full_circ_size] = {'circ':full_circ,'shots':saturated_shots[0],
                        'sv':ground_truths[0],'qasm':saturated_probs[0]}
                    else:
                        print('Use currently running %d qubit full circuit'%full_circ_size)
            print('-'*100)
    print('{:d} cases, {:d} full circuits to run : {}'.format(len(cases_to_run),len(full_circ_to_run),cases_to_run))
    
    scheduler = Scheduler(circ_dict=full_circ_to_run,device_name=args.device_name)
    schedule = scheduler.get_schedule()
    jobs = scheduler.submit_schedule(schedule=schedule)

    # fc_jobs = submit(schedule=fc_schedule,device_name=args.device_name)
    # full_circ_to_run = retrieve(circ_dict=full_circ_to_run,jobs=fc_jobs)
    
    # for full_circ_size in full_circ_sizes:
    #     print('Retrieving %d-qubit circuit'%full_circ_size)
    #     hw_jobs = full_circ_to_run[full_circ_size]['fc_evaluations']['hw']
    #     if 'meas_filter' in full_circ_to_run[full_circ_size]['fc_evaluations']:
    #         meas_filter = full_circ_to_run[full_circ_size]['fc_evaluations']['meas_filter']
    #         del full_circ_to_run[full_circ_size]['fc_evaluations']['meas_filter']
    #     else:
    #         meas_filter = None
    #     hw_prob = accumulate_jobs(jobs=hw_jobs,meas_filter=meas_filter)
    #     full_circ_to_run[full_circ_size]['fc_evaluations']['hw'] = hw_prob
    #     print('*'*50)

    # counter = 1
    # for case in feasible_cases:
    #     print('Case {}'.format(case))
    #     case_dict = {}
    #     cluster_max_qubit,full_circuit_size = case
    #     full_circ = full_circ_to_run[full_circuit_size]['full_circ']
    #     fc_shots = full_circ_to_run[full_circuit_size]['fc_shots']
    #     fc_evaluations = full_circ_to_run[full_circuit_size]['fc_evaluations']
    #     print('%d qubits, %d shots'%(len(full_circ.qubits),fc_shots))
    #     print('sv : %d, qasm : %d, qasm+noise : %d, hw : %d'%(len(fc_evaluations['sv']),len(fc_evaluations['qasm']),len(fc_evaluations['qasm+noise']),len(fc_evaluations['hw'])))
    #     case_dict['full_circ'] = copy.deepcopy(full_circ)
    #     case_dict['fc_shots'] = copy.deepcopy(fc_shots)
    #     case_dict['fc_evaluations'] = copy.deepcopy(fc_evaluations)

    #     searcher_begin = time()
    #     hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
    #     num_clusters=range(2,min(len(full_circ.qubits),args.max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
    #     searcher_time = time() - searcher_begin
    #     if m == None:
    #         print('Case {} NOT feasible'.format(case))
    #     else:
    #         m.print_stat()
    #         clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
    #         case_dict['searcher_time'] = searcher_time
    #         case_dict['clusters'] = copy.deepcopy(clusters)
    #         case_dict['complete_path_map'] = copy.deepcopy(complete_path_map)
    #         pickle.dump({case:case_dict},open(dirname+evaluator_input_filename,'ab'))

    #     print('%d/%d cases'%(counter,total_cases))
    #     counter += 1
    #     print('-'*100)