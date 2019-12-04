import numpy as np
import pickle
import argparse
from qiskit.compiler import transpile, assemble
from utils.helper_fun import get_filename, read_file
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
    job = evaluator_info['device'].run(qobj)
    # job = execute(list(mapped_circuits.values()), backend=Aer.get_backend('qasm_simulator'), shots=batch_shots)
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
    parser = argparse.ArgumentParser(description='Job Submittor')
    parser.add_argument('--experiment-name', metavar='S', type=str,help='which experiment to reconstruct')
    parser.add_argument('--device-name', metavar='S', type=str,help='which device to submit jobs to')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend to use')
    parser.add_argument('--shots-mode', metavar='S', type=str,help='saturated/sametotal shots mode')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']
    assert args.shots_mode in ['saturated','sametotal']
    
    print('-'*50,'Job Submittor','-'*50)

    dirname, job_submittor_input_filename = get_filename(experiment_name=args.experiment_name,
    circuit_type=args.circuit_type,
    device_name=args.device_name,
    evaluation_method=None,field='job_submittor_input')
    job_submittor_input_filename = dirname+job_submittor_input_filename
    job_submittor_input = read_file(filename=job_submittor_input_filename)

    dirname, uniter_input_filename = get_filename(experiment_name=args.experiment_name,
    circuit_type=args.circuit_type,
    device_name=args.device_name,
    evaluation_method=args.evaluation_method,shots_mode=args.shots_mode,field='uniter_input')
    uniter_input_filename = dirname+uniter_input_filename
    uniter_input = read_file(filename=uniter_input_filename)

    print(job_submittor_input.keys())