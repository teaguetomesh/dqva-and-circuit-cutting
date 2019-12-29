import pickle
import os
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from utils.helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots, readout_mitigation, reverseBits, get_filename, read_file, factor_int, schedule_job, cross_entropy
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

def fc_in_dict(case,dictionary):
    for key in dictionary:
        if case[1] == key[1]:
            case_dict = dictionary[key]
            return case_dict
    return {}

def generate_case_dict(case,args,cluster_max_qubit,case_dict):
    if 'full_circ' in case_dict:
        full_circ = case_dict['full_circ']
    else:
        i,j = factor_int(case[1])
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
        case_dict['full_circ'] = full_circ
    searcher_begin = time()
    hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
    num_clusters=range(2,min(len(full_circ.qubits),args.max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
    searcher_time = time() - searcher_begin
    case_dict['searcher_time'] = searcher_time

    if m == None:
        print('Case {} NOT feasible'.format(case))
        return None
    else:
        m.print_stat()
        clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
        case_dict['clusters'] = clusters
        case_dict['complete_path_map'] = complete_path_map
        if 'fc_shots' not in case_dict:
            fc_shots = get_circ_saturated_shots(circs=[full_circ],device_name=args.device_name)[0]
            case_dict['fc_shots'] = fc_shots
        else:
            fc_shots = case_dict['fc_shots']
        if 'fc_evaluations' not in case_dict:
            schedule = schedule_job(circs={'fc':full_circ},shots=fc_shots,max_experiments=device_max_experiments,max_shots=device_max_shots)
            if len(schedule)>10:
                print('Case {} requires {:d} jobs, skipped'.format(case,len(schedule)))
                return None
            else:
                print('saturated fc shots = %d, needs %d jobs'%(case_dict['fc_shots'],len(schedule)))
                fields_to_run = ['sv_noiseless','qasm','hw']
                case_dict['fc_evaluations'] = evaluate_full_circ(circ=full_circ,total_shots=fc_shots,device_name=args.device_name,fields=fields_to_run)
                hw_jobs = case_dict['fc_evaluations']['hw']
                print('Submitting case {} has job ids :'.format(case))
                [print(x['job'].job_id()) for x in hw_jobs]
                if 'meas_filter' in case_dict['fc_evaluations']:
                    meas_filter_job, _, _ = case_dict['fc_evaluations']['meas_filter']
                    print('Meas_filter job id {}'.format(meas_filter_job.job_id()))
        return case_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--min-qubit', metavar='N', type=int,help='Benchmark minimum number of HW qubits')
    parser.add_argument('--max-qubit', metavar='N', type=int,help='Benchmark maximum number of HW qubits')
    parser.add_argument('--max-clusters', metavar='N', type=int,help='max number of clusters to split into')
    parser.add_argument('--device-name', metavar='S',type=str,help='IBM device')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    dirname, evaluator_input_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='evaluator_input',evaluation_method=None,shots_mode=None)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print('-'*50,'Generator','-'*50,flush=True)
    evaluator_input = read_file(dirname+evaluator_input_filename)
    print('Existing cases:',evaluator_input.keys())

    evaluator_info = get_evaluator_info(circ=None,device_name=args.device_name,fields=['properties','device'])
    device_size = len(evaluator_info['properties'].qubits)
    device_max_shots = evaluator_info['device'].configuration().max_shots
    device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/3*2)

    # NOTE: toggle circuits to benchmark
    dimension_l = np.arange(3,16)
    counter = 1
    total_cases = (args.max_qubit-args.min_qubit+1)*len(dimension_l)
    cases_to_run = {}
    for cluster_max_qubit in range(args.min_qubit,args.max_qubit+1):
        for dimension in dimension_l:
            i,j = factor_int(dimension)
            full_circuit_size = i*j
            case = (cluster_max_qubit,full_circuit_size)
            print('-'*100)
            print('Case {}'.format(case))
            if full_circuit_size<=cluster_max_qubit or full_circuit_size>device_size or (cluster_max_qubit-1)*args.max_clusters<full_circuit_size:
                print('Case {} impossible, skipped'.format(case))
            elif case in evaluator_input:
                print('Case {} already exists'.format(case))
            elif fc_in_dict(case=case,dictionary=evaluator_input) != {}:
                print('Use existing full circuit results')
                case_dict = fc_in_dict(case=case,dictionary=evaluator_input)
                case_dict = generate_case_dict(case=case,args=args,cluster_max_qubit=cluster_max_qubit,case_dict=case_dict)
                if case_dict!=None:
                    print('Case {} added to evaluator_input'.format(case))
                    pickle.dump({case:case_dict},open(dirname+evaluator_input_filename,'ab'))
                    evaluator_input = read_file(dirname+evaluator_input_filename)
            elif fc_in_dict(case=case,dictionary=cases_to_run) != {}:
                print('Use currently running full circuit')
                case_dict = fc_in_dict(case=case,dictionary=cases_to_run)
                case_dict = generate_case_dict(case=case,args=args,cluster_max_qubit=cluster_max_qubit,case_dict=case_dict)
                if case_dict!=None:
                    print('Case {} added in cases to run, with currently running full circuit'.format(case))
                    cases_to_run[case] = case_dict
            else:
                print('Generate new circuit')
                case_dict = generate_case_dict(case=case,args=args,cluster_max_qubit=cluster_max_qubit,case_dict={})
                if case_dict!=None:
                    print('Case {} added in cases to run'.format(case))
                    cases_to_run[case] = case_dict
            print('%d/%d cases'%(counter,total_cases))
            counter += 1
            print('-'*100)
    
    counter = 1
    for case in cases_to_run:
        hw_jobs = cases_to_run[case]['fc_evaluations']['hw']
        print('Retrieving case {}'.format(case),flush=True)
        if 'meas_filter' in cases_to_run[case]['fc_evaluations']:
            meas_filter = cases_to_run[case]['fc_evaluations']['meas_filter']
        else:
            meas_filter = None
        execute_begin = time()
        hw_prob = accumulate_jobs(jobs=hw_jobs,meas_filter=meas_filter)
        print('Execute on hardware took %.3e seconds'%(time()-execute_begin))

        case_dict = {}
        for key in cases_to_run[case]:
            if key != 'fc_evaluations':
                case_dict[key] = cases_to_run[case][key]
        case_dict['fc_evaluations'] = {'sv_noiseless':cases_to_run[case]['fc_evaluations']['sv_noiseless'],'qasm':cases_to_run[case]['fc_evaluations']['qasm'],
        'qasm+noise':cases_to_run[case]['fc_evaluations']['qasm+noise'],'hw':hw_prob}
        pickle.dump({case:case_dict},open(dirname+evaluator_input_filename,'ab'))
        qasm_ce = cross_entropy(target=case_dict['fc_evaluations']['sv_noiseless'],obs=case_dict['fc_evaluations']['qasm'])
        hw_ce = cross_entropy(target=case_dict['fc_evaluations']['sv_noiseless'],obs=case_dict['fc_evaluations']['hw'])
        print('qasm \u0394H = %.3e, hw \u0394H = %.3e'%(qasm_ce,hw_ce))
        print('Retrieved %d/%d cases'%(counter,len(cases_to_run)),flush=True)
        counter += 1
        print('*'*50)
    print('-'*100)