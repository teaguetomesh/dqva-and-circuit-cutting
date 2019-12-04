import pickle
import argparse
from utils.helper_fun import get_filename, read_file

def scheduler(job_submittor_input):


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
    print(job_submittor_input[(5,8)]['all_cluster_prob'].keys())