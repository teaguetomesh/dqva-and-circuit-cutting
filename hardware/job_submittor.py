import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from utils.helper_fun import get_evaluator_info, get_circ_saturated_shots, get_filename, read_file, apply_measurement
from utils.submission import Scheduler
from utils.mitigation import TensoredMitigation
from time import time
import copy
from qiskit import Aer
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which device to submit jobs to')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--shots-mode', metavar='S', type=str,help='saturated/sametotal shots mode')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']
    assert args.shots_mode in ['saturated','sametotal']

    dirname, job_submittor_input_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='job_submittor_input',evaluation_method=None,shots_mode=None)
    
    print('-'*50,'Job Submittor','-'*50)
    job_submittor_input = read_file(dirname+job_submittor_input_filename)
    dirname, uniter_input_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='uniter_input',evaluation_method='hardware',shots_mode=args.shots_mode)
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
    mitigation_circ_dict = {}
    for case in cases_to_run:
        print('Case {}'.format(case))
        case_dict = cases_to_run[case]
        for cluster_idx in cases_to_run[case]['all_cluster_prob']:
            cluster_base_circ = case_dict['clusters'][cluster_idx]
            evaluator_info = get_evaluator_info(circ=cluster_base_circ,device_name=args.device_name,
            fields=['device','basis_gates','coupling_map','properties','initial_layout'])
            mitigation_circ_key = '{},{},{}'.format(case[0],case[1],cluster_idx)
            mitigation_circ_dict[mitigation_circ_key] = {'circ':cluster_base_circ,'initial_layout':evaluator_info['initial_layout']}
            if args.shots_mode == 'saturated':
                cluster_shots = get_circ_saturated_shots(circs=[cluster_base_circ],device_name=args.device_name)[0][0]
                print('Cluster %d saturated shots = %d'%(cluster_idx,cluster_shots))
            else:
                fc_shots = case_dict['fc_shots']
                cluster_shots = int(fc_shots/len(case_dict['all_cluster_prob'][cluster_idx]))
                print('Cluster %d sametotal shots = %d'%(cluster_idx,cluster_shots))
            for init_meas in case_dict['all_cluster_prob'][cluster_idx]:
                init_str = ','.join(init_meas[0])
                meas_str = ','.join(init_meas[1])
                key = '{},{},{}|{},{}'.format(case[0],case[1],cluster_idx,init_str,meas_str)
                circ = case_dict['all_cluster_prob'][cluster_idx][init_meas]
                qc=apply_measurement(circ)
                mapped_circuit = transpile(qc,
                backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
                coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
                initial_layout=evaluator_info['initial_layout'])
                circ_dict[key] = {'circ':mapped_circuit,'shots':cluster_shots}

    scheduler = Scheduler(circ_dict=circ_dict,device_name=args.device_name)
    scheduler.run(real_device=False)
    tensored_mitigation = TensoredMitigation(circ_dict=mitigation_circ_dict,device_name=args.device_name)
    tensored_mitigation.run(real_device=False)

    scheduler.retrieve(force_prob=True)
    tensored_mitigation.retrieve()
    
    mitigated = {}
    unmitigated = scheduler.circ_dict
    for mitigation_circ_key in tensored_mitigation.circ_dict:
        calibration_matrix = tensored_mitigation.circ_dict[mitigation_circ_key]['calibration_matrix']
        filter_matrix = np.linalg.inv(calibration_matrix)
        for key in unmitigated:
            if mitigation_circ_key == key.split('|')[0]:
                mitigated[key] = copy.deepcopy(unmitigated[key])
                unmitigated_prob = np.reshape(unmitigated[key]['hw'],(-1,1))
                mitigated_prob = np.reshape(filter_matrix.dot(unmitigated_prob),(1,-1)).tolist()[0]
                assert abs(sum(mitigated_prob)-1)<1e-10
                mitigated[key]['mitigated_hw'] = copy.deepcopy(mitigated_prob)
    circ_dict = copy.deepcopy(mitigated)

    for case in cases_to_run:
        case_dict = cases_to_run[case]
        case_dict['mitigated_all_cluster_prob'] = {}
        for cluster_idx in case_dict['all_cluster_prob']:
            case_dict['mitigated_all_cluster_prob'][cluster_idx] = {}
            for init_meas in case_dict['all_cluster_prob'][cluster_idx]:
                init_str = ','.join(init_meas[0])
                meas_str = ','.join(init_meas[1])
                key = '{},{},{}|{},{}'.format(case[0],case[1],cluster_idx,init_str,meas_str)
                case_dict['all_cluster_prob'][cluster_idx][init_meas] = copy.deepcopy(circ_dict[key]['hw'])
                case_dict['mitigated_all_cluster_prob'][cluster_idx][init_meas] = copy.deepcopy(circ_dict[key]['mitigated_hw'])
        pickle.dump({case:case_dict}, open(dirname+uniter_input_filename,'ab'))