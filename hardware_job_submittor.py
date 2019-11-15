import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from helper_fun import get_evaluator_info, apply_measurement, reverseBits, get_circ_saturated_shots
from time import time
import copy
from qiskit import Aer

def update_counts(cumulated, batch):
    for state in batch:
        if state not in cumulated:
            cumulated[state] = batch[state]
        else:
            cumulated[state] = cumulated[state] + batch[state]
    return cumulated

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

    hw_counts = {}
    for init_meas in mapped_circuits:
        hw_counts[init_meas] = {}
    
    # FIXME: split up hardware shots
    device_max_shots = evaluator_info['device'].configuration().max_shots
    remaining_shots = evaluator_info['num_shots']
    while remaining_shots>0:
        batch_shots = min(remaining_shots,device_max_shots)
        print('Submitted %d circuits to hardware, %d shots'%(len(cluster_instances),batch_shots))
        qobj = assemble(list(mapped_circuits.values()), backend=evaluator_info['device'], shots=batch_shots)
        job = evaluator_info['device'].run(qobj)
        hw_results = job.result()

        if 'meas_filter' in evaluator_info:
            print('Mitigation for %d * %d-qubit circuit'%(len(cluster_instances),len(circ.qubits)))
            mitigation_begin = time()
            mitigated_results = evaluator_info['meas_filter'].apply(hw_results)
            mitigation_time = time() - mitigation_begin
            print('Mitigation for %d * %d-qubit circuit took %.3e seconds'%(len(cluster_instances),len(circ.qubits),mitigation_time))
            for init_meas in mapped_circuits:
                hw_count = mitigated_results.get_counts(mapped_circuits[init_meas])
                hw_counts[init_meas] = update_counts(cumulated=hw_counts[init_meas], batch=hw_count)
        else:
            for init_meas in mapped_circuits:
                hw_count = hw_results.get_counts(mapped_circuits[init_meas])
                # print('batch {} counts:'.format(init_meas),hw_count)
                # print('cumulative {} counts:'.format(init_meas),hw_counts[init_meas])
                hw_counts[init_meas] = update_counts(cumulated=hw_counts[init_meas], batch=hw_count)
        remaining_shots -= batch_shots
    
    hw_probs = {}
    for init_meas in hw_counts:
        circ = cluster_instances[init_meas]
        hw_count = hw_counts[init_meas]
        hw_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in hw_count:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            hw_prob[reversed_state] = hw_count[state]/evaluator_info['num_shots']
        hw_probs[init_meas] = hw_prob
    return hw_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which device to submit jobs to')
    parser.add_argument('--circuit-name', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--saturated-shots',action="store_true",help='run saturated number of cluster shots?')
    args = parser.parse_args()

    device_name = args.device_name
    circuit_type = args.circuit_name
    print(device_name)
    input_file = './benchmark_data/job_submittor_input_{}_{}'.format(device_name,circuit_type)
    if args.saturated_shots:
        input_file = input_file+'_saturated.p'
    else:
        input_file = input_file+'_sametotal.p'

    job_submittor_input = pickle.load(open(input_file, 'rb' ))
    job_submittor_output = {}
    filename = input_file.replace('job_submittor_input','hardware_uniter_input')

    for case in job_submittor_input:
        print('Case ',case)
        job_submittor_output[case] = copy.deepcopy(job_submittor_input[case])
        total_shots = job_submittor_input[case]['total_shots']
        for cluster_idx, cluster_circ in enumerate(job_submittor_input[case]['clusters']):
            cluster_instances = job_submittor_input[case]['all_cluster_prob'][cluster_idx]
            print('Cluster %d has %d instances'%(cluster_idx,len(cluster_instances)))
            evaluator_info = get_evaluator_info(circ=cluster_circ,device_name=device_name,
            fields=['device','basis_gates','coupling_map','properties','initial_layout','meas_filter'])
            if args.saturated_shots:
                evaluator_info['num_shots'] = get_circ_saturated_shots(circ=cluster_circ,accuracy=1e-1)
            else:
                evaluator_info['num_shots'] = int(total_shots/len(cluster_instances))+1
            max_experiments = int(evaluator_info['device'].configuration().max_experiments/2)
            hw_begin = time()
            hw_probs = {}
            cluster_instances_batch = {}
            for init_meas in cluster_instances:
                if len(cluster_instances_batch)==max_experiments:
                    hw_probs_batch = submit_hardware_jobs(cluster_instances=cluster_instances_batch,evaluator_info=evaluator_info)
                    hw_probs_batch_copy = copy.deepcopy(hw_probs_batch)
                    hw_probs.update(hw_probs_batch_copy)
                    cluster_instances_batch = {}
                    cluster_instances_batch[init_meas] = cluster_instances[init_meas]
                else:
                    cluster_instances_batch[init_meas] = cluster_instances[init_meas]
            hw_probs_batch = submit_hardware_jobs(cluster_instances=cluster_instances_batch,evaluator_info=evaluator_info)
            hw_probs_batch_copy = copy.deepcopy(hw_probs_batch)
            hw_probs.update(hw_probs_batch_copy)
            hw_elapsed = time()-hw_begin
            print('Hardware queue time = %.3e seconds'%hw_elapsed)
            job_submittor_output[case]['all_cluster_prob'][cluster_idx] = copy.deepcopy(hw_probs)
        try:
            curr_job_submittor_output = pickle.load(open('%s'%filename, 'rb' ))
        except:
            curr_job_submittor_output = {}
        curr_job_submittor_output[case] = copy.deepcopy(job_submittor_output[case])
        pickle.dump(curr_job_submittor_output, open('%s'%filename,'wb'))
        print('Job submittor output has %d cases'%len(curr_job_submittor_output))
        print('*'*50)
    print('-'*100)