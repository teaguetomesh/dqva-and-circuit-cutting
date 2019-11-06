import pickle
import argparse
from helper_fun import get_evaluator_info, evaluate_circ, apply_measurement

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
            fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model','num_shots'])
            cluster_instances = job_submittor_input[case]['all_cluster_prob'][cluster_idx]
            print('Cluster %d has %d instances'%(cluster_idx,len(cluster_instances)))
            for key in cluster_instances:
                # FIXME: debug here
                # qc = apply_measurement(cluster_instances[key])
                hw_fc = evaluate_circ(circ=cluster_instances[key],backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
                cluster_instances[key] = hw_fc

            job_submittor_input[case]['all_cluster_prob'][cluster_idx] = cluster_instances
    filename = args.input_file.replace('job_submittor_input','hardware_uniter_input')
    pickle.dump(job_submittor_input, open('%s'%filename,'wb'))