import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from utils.helper_fun import get_evaluator_info, get_circ_saturated_shots, get_filename, read_file
from utils.schedule import Scheduler
from utils.conversions import reverse_prob
from time import time
import copy
from qiskit import Aer
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which device to submit jobs to')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    dirname, job_submittor_input_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='job_submittor_input',evaluation_method=None)
    
    print('-'*50,'Job Submittor','-'*50)
    job_submittor_input = read_file(dirname+job_submittor_input_filename)
    dirname, uniter_input_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='uniter_input',evaluation_method='hardware')
    job_submittor_output = read_file(dirname+uniter_input_filename)
    print('Existing cases:',job_submittor_output.keys())
    cases_to_run = {}
    for case in job_submittor_input:
        if case not in job_submittor_output:
            cases_to_run[case] = copy.deepcopy(job_submittor_input[case])
    print('Run cases:',cases_to_run.keys())
    print('*'*50)

    evaluator_info = get_evaluator_info(circ=None,device_name=args.device_name,fields=['properties','device'])
    device_max_shots = evaluator_info['device'].configuration().max_shots
    device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/3*2)

    circ_dict = {}
    for case in cases_to_run:
        print('Case {}'.format(case))
        case_dict = cases_to_run[case]
        for cluster_idx in cases_to_run[case]['all_cluster_prob']:
            cluster_base_circ = case_dict['clusters'][cluster_idx]
            evaluator_info = get_evaluator_info(circ=cluster_base_circ,device_name=args.device_name,
            fields=['basis_gates','coupling_map','properties','initial_layout'])
            backend_device = get_evaluator_info(circ=None,device_name=args.device_name,fields=['device'])['device']
            fc_shots = case_dict['fc_shots']
            cluster_shots = int(fc_shots/len(case_dict['all_cluster_prob'][cluster_idx]))
            cluster_shots = max(cluster_shots,1024)
            print('Cluster %d sametotal shots = %d'%(cluster_idx,cluster_shots))
            for init_meas in case_dict['all_cluster_prob'][cluster_idx]:
                init_str = ','.join(init_meas[0])
                meas_str = ','.join(init_meas[1])
                key = '{},{},{}|{},{}'.format(case[0],case[1],cluster_idx,init_str,meas_str)
                circ = case_dict['all_cluster_prob'][cluster_idx][init_meas]
                circ_dict[key] = {'circ':circ,'shots':cluster_shots,'initial_layout':evaluator_info['initial_layout']}

    scheduler = Scheduler(circ_dict=circ_dict,device_name=args.device_name)
    scheduler.run(real_device=False)
    scheduler.retrieve(force_prob=True)
    circ_dict = scheduler.circ_dict

    for case in cases_to_run:
        case_dict = cases_to_run[case]
        for cluster_idx in case_dict['all_cluster_prob']:
            for init_meas in case_dict['all_cluster_prob'][cluster_idx]:
                init_str = ','.join(init_meas[0])
                meas_str = ','.join(init_meas[1])
                key = '{},{},{}|{},{}'.format(case[0],case[1],cluster_idx,init_str,meas_str)
                case_dict['all_cluster_prob'][cluster_idx][init_meas] = copy.deepcopy(reverse_prob(prob_l=circ_dict[key]['hw']))
        pickle.dump({case:case_dict}, open(dirname+uniter_input_filename,'ab'))