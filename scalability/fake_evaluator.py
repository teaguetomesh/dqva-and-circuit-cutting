import pickle
import itertools
import copy
import numpy as np
from time import time
import argparse
from utils.helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots, distribute_cluster_shots, get_filename, read_file
from utils.conversions import dict_to_array, reverse_prob

def find_all_simulation_combinations(O_qubits, rho_qubits, d_qubits):
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','minus','plus_i','minus_i']
    all_inits = list(itertools.product(init_states,repeat=rho_qubits))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(d_qubits)]
        for i in range(len(init)):
            complete_init[i] = init[i]
        complete_inits.append(complete_init)
    # print('initializations:',complete_inits)

    # print('O qubits:',O_qubits)
    all_meas = list(itertools.product(measurement_basis,repeat=O_qubits))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['I' for i in range(d_qubits)]
        for i in range(len(meas)):
            complete_m[i] = meas[i]
        complete_meas.append(complete_m)
    # print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    return combinations

def evaluate_cluster(combinations,num_qubits):
    cluster_prob = {}
    uniform_p = 1/2**num_qubits
    # uniform_prob = np.array([uniform_p for x in range(2**num_qubits)])
    for _, combination in enumerate(combinations):
        inits, meas = combination
        cluster_prob[(tuple(inits),tuple(meas))] = 2**num_qubits
    return cluster_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    print('-'*50,'Evaluator','-'*50,flush=True)
    dirname, evaluator_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,device_name='fake',field='evaluator_input',evaluation_method='fake')
    evaluator_input = read_file(dirname+evaluator_input_filename)

    dirname, uniter_input_filename = get_filename(experiment_name='scalability',circuit_type=args.circuit_type,device_name='fake',field='uniter_input',evaluation_method='fake')
    output_filename = dirname + uniter_input_filename
    evaluator_output = read_file(output_filename)

    print('Existing cases:',evaluator_output.keys())
    counter = len(evaluator_output.keys())
    for case in evaluator_input:
        if case in evaluator_output:
            continue
        else:
            print('Running case:',case,flush=True)
            case_dict = copy.deepcopy(evaluator_input[case])
            num_d_qubits = case_dict['num_d_qubits']
            num_rho_qubits = case_dict['num_rho_qubits']
            num_O_qubits = case_dict['num_O_qubits']
            case_dict['all_cluster_prob'] = {}
            for cluster_idx in range(len(num_d_qubits)):
                d_qubits = num_d_qubits[cluster_idx]
                rho_qubits = num_rho_qubits[cluster_idx]
                O_qubits = num_O_qubits[cluster_idx]
                combinations = find_all_simulation_combinations(O_qubits, rho_qubits, d_qubits)
                print('Case {}, cluster_{:d} {:d}_qubits * {:d}_instances on fake QUANTUM SIMULATOR, '.format(case,cluster_idx,d_qubits,len(combinations)))
                cluster_prob = evaluate_cluster(combinations=combinations,num_qubits=d_qubits)
                case_dict['all_cluster_prob'][cluster_idx] = cluster_prob
            pickle.dump({case:case_dict}, open(output_filename,'ab'))
            counter += 1
            print('dumped case {}, {:d}/{:d} cases'.format(case,counter,len(evaluator_input)))
            print('-'*100)