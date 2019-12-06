import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from utils.helper_fun import get_evaluator_info, apply_measurement, reverseBits, get_circ_saturated_shots, distribute_cluster_shots, readout_mitigation, get_filename, read_file
from time import time
import copy
from qiskit import Aer, execute
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

def update_counts(cumulated, batch):
    for state in batch:
        if state not in cumulated:
            cumulated[state] = batch[state]
        else:
            cumulated[state] = cumulated[state] + batch[state]
    return cumulated

def split_cluster_instances(circs,shots,max_experiments,max_shots):
    if len(circs)<=max_experiments and shots<=max_shots:
        current_schedule = [(circs,shots)]
        return current_schedule
    elif len(circs)>max_experiments and shots<=max_shots:
        current_instances = {}
        next_instances = {}
        for init_meas in circs:
            if len(current_instances) < max_experiments:
                current_instances[init_meas] = circs[init_meas]
            else:
                next_instances[init_meas] = circs[init_meas]
        current_schedule = [(current_instances,shots)]
        next_schedule = split_cluster_instances(circs=next_instances,shots=shots,max_experiments=max_experiments,max_shots=max_shots)
        return current_schedule + next_schedule
    elif len(circs)<=max_experiments and shots>max_shots:
        current_schedule = [(circs,max_shots)]
        next_schedule = split_cluster_instances(circs=circs,shots=shots-max_shots,max_experiments=max_experiments,max_shots=max_shots)
        return current_schedule + next_schedule
    elif len(circs)>max_experiments and shots>max_shots:
        left_circs = {}
        right_circs = {}
        for init_meas in circs:
            if len(left_circs) < max_experiments:
                left_circs[init_meas] = circs[init_meas]
            else:
                right_circs[init_meas] = circs[init_meas]
        left_schedule = split_cluster_instances(circs=left_circs,shots=shots,max_experiments=max_experiments,max_shots=max_shots)
        right_schedule = split_cluster_instances(circs=right_circs,shots=shots,max_experiments=max_experiments,max_shots=max_shots)
        return left_schedule + right_schedule
    else:
        raise Exception('This condition should not happen')

def submit_hardware_jobs(cluster_instances, evaluator_info):
    mapped_circuits = {}
    for init_meas in cluster_instances:
        circ = cluster_instances[init_meas]
        qc=apply_measurement(circ)
        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])
        mapped_circuits[init_meas] = mapped_circuit

    qobj = assemble(list(mapped_circuits.values()), backend=evaluator_info['device'], shots=batch_shots)
    # job = evaluator_info['device'].run(qobj)
    job = Aer.get_backend('qasm_simulator').run(qobj)
    job_dict = {'job':job,'mapped_circuits':mapped_circuits,'evaluator_info':evaluator_info}
    return job_dict

def accumulate_cluster_jobs(cluster_job_dict,cluster_meas_filter):
    if cluster_meas_filter != None:
        meas_filter_job, state_labels, qubit_list = cluster_meas_filter
        print('Meas filter job id {}'.format(meas_filter_job.job_id()))
        cal_results = meas_filter_job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
        meas_filter = meas_fitter.filter
        num_qubits_mitigated = len(qubit_list)
    else:
        num_qubits_mitigated = -1
    hw_counts = {}
    for job_dict in cluster_job_dict:
        mapped_circuits = job_dict['mapped_circuits']
        job = job_dict['job']
        print(job.job_id())
        evaluator_info = job_dict['evaluator_info']
        hw_results = job.result()
        mapped_circuit = list(mapped_circuits.values())[0]

        if cluster_meas_filter != None:
            print('Mitigation for %d * %d-qubit circuit'%(len(mapped_circuits),num_qubits_mitigated))
            mitigation_begin = time()
            hw_results = meas_filter.apply(hw_results)
            mitigation_time = time() - mitigation_begin
            print('Mitigation for %d * %d-qubit circuit took %.3e seconds'%(len(mapped_circuits),num_qubits_mitigated,mitigation_time))
        for init_meas in mapped_circuits:
            hw_count = hw_results.get_counts(mapped_circuits[init_meas])
            try:
                hw_counts[init_meas] = update_counts(cumulated=hw_counts[init_meas], batch=hw_count)
            except:
                hw_counts[init_meas] = update_counts(cumulated={}, batch=hw_count)
    
    hw_probs = {}
    for init_meas in hw_counts:
        hw_count = hw_counts[init_meas]
        num_qubits = len(list(hw_count.keys())[0])
        num_shots = sum(hw_count.values())
        hw_prob = [0 for x in range(np.power(2,num_qubits))]
        for state in hw_count:
            reversed_state = reverseBits(int(state,2),num_qubits)
            hw_prob[reversed_state] = hw_count[state]/num_shots
        hw_probs[init_meas] = hw_prob
    return hw_probs

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
    device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/2)
    
    all_submitted_jobs = {}
    for case in cases_to_run:
        print('submitting case ',case)
        all_submitted_jobs[case] = {}
        fc_shots = cases_to_run[case]['fc_shots']
        clusters = cases_to_run[case]['clusters']
        complete_path_map = cases_to_run[case]['complete_path_map']
        same_total_cutting_shots = distribute_cluster_shots(total_shots=fc_shots,clusters=clusters,complete_path_map=complete_path_map)

        for cluster_idx, cluster_circ in enumerate(clusters):
            cluster_instances = cases_to_run[case]['all_cluster_prob'][cluster_idx]
            evaluator_info = get_evaluator_info(circ=cluster_circ,device_name=args.device_name,
            fields=['device','basis_gates','coupling_map','properties','initial_layout'])

            if args.shots_mode == 'saturated':
                evaluator_info['num_shots'] = get_circ_saturated_shots(circs=[cluster_circ],accuracy=1e-1)[0]
            elif args.shots_mode == 'sametotal':
                evaluator_info['num_shots'] = same_total_cutting_shots[cluster_idx]
            print('Cluster %d has %d instances, %d shots each, submission history:'%(cluster_idx,len(cluster_instances),evaluator_info['num_shots']))

            all_submitted_jobs[case][cluster_idx] = {'jobs':[],'meas_filter':None}
            if np.power(2,len(cluster_circ.qubits))<=device_max_experiments:
                meas_filter_job, state_labels, qubit_list = readout_mitigation(device=evaluator_info['device'],initial_layout=evaluator_info['initial_layout'])
                all_submitted_jobs[case][cluster_idx]['meas_filter'] = (meas_filter_job, state_labels, qubit_list)
                print('Meas_filter job id {}'.format(meas_filter_job.job_id()))

            schedule = split_cluster_instances(circs=cluster_instances,shots=evaluator_info['num_shots'],max_experiments=device_max_experiments,max_shots=device_max_shots)

            for s in schedule:
                cluster_instances_batch, batch_shots = s
                evaluator_info['num_shots'] = batch_shots
                job_dict = submit_hardware_jobs(cluster_instances=cluster_instances_batch, evaluator_info=evaluator_info)
                print('Submitted %d circuits to hardware, %d shots. job_id ='%(len(cluster_instances_batch),evaluator_info['num_shots']),job_dict['job'].job_id())
                all_submitted_jobs[case][cluster_idx]['jobs'].append(job_dict)
        print('*'*50)
    print('-'*100)
            
    for case in all_submitted_jobs:
        for cluster_idx in all_submitted_jobs[case]:
            cluster_job_dict = all_submitted_jobs[case][cluster_idx]['jobs']
            print('Retrieving case {} cluster {:d} has {:d} jobs'.format(case,cluster_idx,len(cluster_job_dict)))
            hw_probs = accumulate_cluster_jobs(cluster_job_dict=cluster_job_dict,cluster_meas_filter=all_submitted_jobs[case][cluster_idx]['meas_filter'])
            cases_to_run[case]['all_cluster_prob'][cluster_idx] = hw_probs
        case_dict = cases_to_run[case]
        pickle.dump({case:case_dict}, open(dirname+uniter_input_filename,'ab'))
        print('Job submittor output has %d cases'%len(job_submittor_output))
        print('*'*50)
    print('-'*100)