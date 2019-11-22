import pickle
import os
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
import MIQCP_searcher as searcher
import cutter
from helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots, readout_mitigation, reverseBits
import argparse
from qiskit import IBMQ
import copy
import math
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

def gen_secret(num_qubit):
    num_digit = num_qubit-1
    num = 2**num_digit-1
    num = bin(num)[2:]
    num_with_zeros = str(num).zfill(num_digit)
    return num_with_zeros

def accumulate_jobs(jobs,meas_filter):
    hw_counts = {}
    if meas_filter != None:
        meas_filter_job, state_labels, qubit_list = meas_filter
        print('Meas filter job id {}'.format(meas_filter_job.job_id()))
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
            print('Mitigation for %d qubit circuit'%(len(circ.qubits)))
            mitigation_begin = time()
            hw_result = meas_filter.apply(hw_result)
            print('Mitigation for %d qubit circuit took %.3e seconds'%(len(circ.qubits),time()-mitigation_begin))
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

    if 'sv_noiseless' in fields:
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
        print('Evaluating fc hardware, %d shots, submission history:'%total_shots)
        hw_evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
        fields=['device','basis_gates','coupling_map','properties','initial_layout'])
        hw_evaluator_info['num_shots'] = total_shots
        submission_begin = time()
        hw_jobs = evaluate_circ(circ=circ,backend='hardware',evaluator_info=hw_evaluator_info)
        print('job object turnaround time =',time()-submission_begin)
        if np.power(2,len(circ.qubits))<hw_evaluator_info['device'].configuration().max_experiments/3*2:
            submission_begin = time()
            meas_filter_job, state_labels, qubit_list = readout_mitigation(device=hw_evaluator_info['device'],initial_layout=hw_evaluator_info['initial_layout'])
            print('job object turnaround time =',time()-submission_begin)
            fc_evaluations['meas_filter'] = (meas_filter_job, state_labels, qubit_list)
    else:
        hw_jobs = uniform_prob

    fc_evaluations.update({'sv_noiseless':sv_noiseless_fc,
    'qasm':qasm_noiseless_fc,
    'qasm+noise':qasm_noisy_fc,
    'hw':hw_jobs})

    return fc_evaluations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--min-qubit', metavar='N', type=int,help='Benchmark minimum number of HW qubits')
    parser.add_argument('--max-qubit', metavar='N', type=int,help='Benchmark maximum number of HW qubits')
    parser.add_argument('--max-clusters', metavar='N', type=int,help='max number of clusters to split into')
    parser.add_argument('--device-name', metavar='S',type=str,help='IBM device')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    print('-'*50,'Generator %s %s'%(args.device_name,args.circuit_type),'-'*50)
    
    dirname = './benchmark_data'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    dirname = './benchmark_data/%s'%args.circuit_type
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    try:
        f = open('./benchmark_data/evaluator_input_{}_{}.p'.format(args.device_name,args.circuit_type),'rb')
        evaluator_input = pickle.load(f)
        print('Existing cases:',evaluator_input.keys())
    except:
        evaluator_input = {}

    evaluator_info = get_evaluator_info(circ=None,device_name=args.device_name,fields=['properties','device'])
    device_size = len(evaluator_info['properties'].qubits)
    device_max_shots = evaluator_info['device'].configuration().max_shots
    device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/2)

    # NOTE: toggle circuits to benchmark
    dimension_l = [[1,3],[2,2],[1,5],[2,3],[1,7],[2,4],[3,3],[2,5],[3,4],[2,7],[4,4],[3,6],[4,5]]
    full_circs = {}
    cases_to_run = {}
    for cluster_max_qubit in range(args.min_qubit,args.max_qubit+1):
        for dimension in dimension_l:
            i,j = dimension
            full_circuit_size = i*j
            if full_circuit_size<=cluster_max_qubit or full_circuit_size>device_size or (cluster_max_qubit-1)*args.max_clusters<full_circuit_size:
                continue
            
            case = (cluster_max_qubit,full_circuit_size)
            if case not in [(4,8)]:
                continue
            if case in evaluator_input:
                continue
            
            print('-'*100)
            print('Case',case)

            if full_circuit_size in full_circs:
                print('Use existing full circuit')
                full_circ = full_circs[full_circuit_size]
            else:
                if args.circuit_type == 'supremacy':
                    full_circ = gen_supremacy(i,j,8)
                elif args.circuit_type == 'hwea':
                    full_circ = gen_hwea(i*j,1)
                elif args.circuit_type == 'bv':
                    full_circ = gen_BV(gen_secret(i*j),barriers=False)
                elif args.circuit_type == 'qft':
                    full_circ = gen_qft(width=i*j, barriers=False)
                elif args.circuit_type == 'sycamore':
                    full_circ = gen_sycamore(i,j,8)
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
                fc_shots = get_circ_saturated_shots(circs=[full_circ],accuracy=1e-2)[0]
                num_jobs = math.ceil(fc_shots/device_max_shots/device_max_experiments)
                if num_jobs>10:
                    print('Case {} needs {} jobs'.format(case,num_jobs))
                    print('-'*100)
                    continue
                print('saturated fc shots = %d, needs %d jobs'%(fc_shots,num_jobs))
                case_dict = {'full_circ':full_circ,'fc_shots':fc_shots,'searcher_time':searcher_time,
                'clusters':clusters,'complete_path_map':complete_path_map}
                cases_to_run[case] = copy.deepcopy(case_dict)
                print('%d cases to run:'%(len(cases_to_run)),cases_to_run.keys())
                print('-'*100)
    for case in cases_to_run:
        fc_jobs = math.ceil(cases_to_run[case]['fc_shots']/device_max_shots/device_max_experiments)
        print('case {} needs {} fc jobs'.format(case,fc_jobs))
    print('-'*100)

    fields_to_run = ['sv_noiseless','qasm','hw']
    for case in cases_to_run:
        full_circ = cases_to_run[case]['full_circ']
        fc_shots = cases_to_run[case]['fc_shots']
        fc_evaluations = evaluate_full_circ(circ=full_circ,total_shots=fc_shots,device_name=args.device_name,fields=fields_to_run)
        cases_to_run[case]['fc_evaluations'] = fc_evaluations
        hw_jobs = fc_evaluations['hw']
        print('Submitting case {} has job ids {}'.format(case,[x['job'].job_id() for x in hw_jobs]))
        try:
            meas_filter_job, _, _ = fc_evaluations['meas_filter']
            print('Meas_filter job id {}'.format(meas_filter_job.job_id()))
        except:
            meas_filter_job = None
        print('*'*50)
    print('Submitted %d cases to hw'%(len(cases_to_run)))
    print('-'*100)
    for case in cases_to_run:
        if 'hw' not in fields_to_run:
            case_dict = copy.deepcopy(cases_to_run[case])
            evaluator_input.update({case:case_dict})
        else:
            hw_jobs = cases_to_run[case]['fc_evaluations']['hw']
            print('Retrieving case {}'.format(case))
            try:
                meas_filter = cases_to_run[case]['fc_evaluations']['meas_filter']
                meas_filter_job, _, _ = meas_filter
            except:
                meas_filter = None
            execute_begin = time()
            hw_prob = accumulate_jobs(jobs=hw_jobs,meas_filter=meas_filter)
            print('Execute on hardware took %.3e seconds'%(time()-execute_begin))
            
            cases_to_run[case]['fc_evaluations'] = {'sv_noiseless':cases_to_run[case]['fc_evaluations']['sv_noiseless'],'qasm':cases_to_run[case]['fc_evaluations']['qasm'],
            'qasm+noise':cases_to_run[case]['fc_evaluations']['qasm+noise'],'hw':hw_prob}

            case_dict = copy.deepcopy(cases_to_run[case])
            evaluator_input.update({case:case_dict})
        print('Dump evaluator_input with %d cases'%(len(evaluator_input)))
        pickle.dump(evaluator_input,open('./benchmark_data/evaluator_input_{}_{}.p'.format(args.device_name,args.circuit_type),'wb'))
    print('-'*100)