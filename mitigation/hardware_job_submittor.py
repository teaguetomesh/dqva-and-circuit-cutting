import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from helper_fun import get_evaluator_info, apply_measurement, reverseBits, get_circ_saturated_shots, distribute_cluster_shots
from time import time
import copy
from qiskit import Aer
from mpi4py import MPI

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
    print('Submitted %d circuits to hardware, %d shots'%(len(cluster_instances),evaluator_info['num_shots']))
    mapped_circuits = {}
    for init_meas in cluster_instances:
        circ = cluster_instances[init_meas]
        qc=apply_measurement(circ)
        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])
        mapped_circuits[init_meas] = mapped_circuit

    hw_counts = {}
    for init_meas in mapped_circuits:
        hw_counts[init_meas] = {}

    qobj = assemble(list(mapped_circuits.values()), backend=evaluator_info['device'], shots=batch_shots)
    # job = evaluator_info['device'].run(qobj)
    job = Aer.get_backend('qasm_simulator').run(qobj)
    job_dict = {'job':job,'mapped_circuits':mapped_circuits,'evaluator_info':evaluator_info}
    return job_dict

    # hw_results = job.result()

    # if 'meas_filter' in evaluator_info:
    #     print('Mitigation for %d * %d-qubit circuit'%(len(cluster_instances),len(circ.qubits)))
    #     mitigation_begin = time()
    #     mitigated_results = evaluator_info['meas_filter'].apply(hw_results)
    #     mitigation_time = time() - mitigation_begin
    #     print('Mitigation for %d * %d-qubit circuit took %.3e seconds'%(len(cluster_instances),len(circ.qubits),mitigation_time))
    #     for init_meas in mapped_circuits:
    #         hw_count = mitigated_results.get_counts(mapped_circuits[init_meas])
    #         hw_counts[init_meas] = update_counts(cumulated=hw_counts[init_meas], batch=hw_count)
    # else:
    #     for init_meas in mapped_circuits:
    #         hw_count = hw_results.get_counts(mapped_circuits[init_meas])
    #         # print('batch {} counts:'.format(init_meas),hw_count)
    #         # print('cumulative {} counts:'.format(init_meas),hw_counts[init_meas])
    #         hw_counts[init_meas] = update_counts(cumulated=hw_counts[init_meas], batch=hw_count)
    # remaining_shots -= batch_shots
    
    # hw_probs = {}
    # for init_meas in hw_counts:
    #     circ = cluster_instances[init_meas]
    #     hw_count = hw_counts[init_meas]
    #     hw_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
    #     for state in hw_count:
    #         reversed_state = reverseBits(int(state,2),len(circ.qubits))
    #         hw_prob[reversed_state] = hw_count[state]/evaluator_info['num_shots']
    #     hw_probs[init_meas] = hw_prob
    # return hw_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which device to submit jobs to')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--shots-mode', metavar='S', type=str,help='saturated/sametotal shots mode')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']
    assert args.shots_mode in ['saturated','sametotal']
    
    print('-'*50,'Job Submittor %s %s %s'%(args.shots_mode,args.circuit_type,args.device_name),'-'*50)
    input_file = './benchmark_data/{}/job_submittor_input_{}_{}_{}.p'.format(args.circuit_type,args.device_name,args.circuit_type,args.shots_mode)
    job_submittor_input = pickle.load(open(input_file, 'rb' ))
    cases_to_run = {}
    filename = input_file.replace('job_submittor_input','hardware_uniter_input')
    try:
        f = open('%s'%filename,'rb')
        job_submittor_output = pickle.load(f)
        print('Existing cases:',job_submittor_output.keys())
    except:
        job_submittor_output = {}
    for case in job_submittor_input:
        if case not in job_submittor_output:
            cases_to_run[case] = job_submittor_input[case]
    print('Run cases:',cases_to_run.keys())
    print('*'*50)
    
    for case in cases_to_run:
        print('submitting case ',case)
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

            # if np.power(2,len(cluster_circ.qubits))<=max_experiments:
            #     _evaluator_info = get_evaluator_info(circ=cluster_circ,device_name=args.device_name,fields=['meas_filter'])
            #     evaluator_info.update(_evaluator_info)
            
            device_max_shots = evaluator_info['device'].configuration().max_shots
            device_max_experiments = int(evaluator_info['device'].configuration().max_experiments/3*2)

            schedule = split_cluster_instances(circs=cluster_instances,shots=evaluator_info['num_shots'],max_experiments=device_max_experiments,max_shots=device_max_shots)
            for s in schedule:
                cluster_instances_batch, batch_shots = s
                evaluator_info['num_shots'] = batch_shots
                job_dict = submit_hardware_jobs(cluster_instances=cluster_instances_batch, evaluator_info=evaluator_info)
        print('*'*50)