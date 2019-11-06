import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from helper_fun import get_evaluator_info, evaluate_circ, apply_measurement, reverseBits

def submit_hardware_jobs(cluster_instances, evaluator_info):
    if evaluator_info['num_shots']>evaluator_info['device'].configuration().max_shots:
        print('During circuit evaluation on hardware, num_shots %.3e exceeded hardware max'%evaluator_info['num_shots'])
        evaluator_info['num_shots'] = evaluator_info['device'].configuration().max_shots
    
    mapped_circuits = {}
    for init_meas in cluster_instances:
        circ = cluster_instances[init_meas]
        qc=apply_measurement(circ)
        mapped_circuit = transpile(qc,
        backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
        coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
        initial_layout=evaluator_info['initial_layout'])
        mapped_circuits[init_meas] = mapped_circuit

    qobj = assemble(list(mapped_circuits.values()), backend=evaluator_info['device'], shots=evaluator_info['num_shots'])
    job = evaluator_info['device'].run(qobj)
    hw_results = job.result()

    hw_counts = {}
    if 'meas_filter' in evaluator_info:
        mitigated_results = evaluator_info['meas_filter'].apply(hw_results)
        for init_meas in mapped_circuits:
            hw_count = mitigated_results.get_counts(mapped_circuits[init_meas])
            hw_counts[init_meas] = hw_count
    else:
        for init_meas in mapped_circuits:
            hw_count = hw_results.get_counts(mapped_circuits[init_meas])
            hw_counts[init_meas] = hw_count
    
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
    parser.add_argument('--input-file', metavar='S', type=str,help='which evaluator input file to run')
    parser.add_argument('--saturated-shots',action="store_true",help='run saturated number of cluster shots')
    args = parser.parse_args()

    s = '_'
    device_name = args.input_file.split('job_submittor_input_')[1].split('.p')[0].split('_')[:-1]
    device_name = s.join(device_name)
    print(device_name)

    job_submittor_input = pickle.load(open(args.input_file, 'rb' ))

    for case in job_submittor_input:
        print('Case ',case)
        print(job_submittor_input[case].keys())
        for cluster_idx, cluster_circ in enumerate(job_submittor_input[case]['clusters']):
            evaluator_info = get_evaluator_info(circ=cluster_circ,device_name=device_name,
            fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model','num_shots','meas_filter'])
            cluster_instances = job_submittor_input[case]['all_cluster_prob'][cluster_idx]
            print('Cluster %d has %d instances'%(cluster_idx,len(cluster_instances)))
            hw_probs = submit_hardware_jobs(cluster_instances=cluster_instances,evaluator_info=evaluator_info)
            job_submittor_input[case]['all_cluster_prob'][cluster_idx] = hw_probs
    filename = args.input_file.replace('job_submittor_input','hardware_uniter_input')
    pickle.dump(job_submittor_input, open('%s'%filename,'wb'))