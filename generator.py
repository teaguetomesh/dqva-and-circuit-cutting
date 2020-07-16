import pickle
import math
import argparse
import os
import subprocess
from termcolor import colored

from utils.MIP_searcher import find_cuts
from utils.cutter import cut_circuit
from utils.helper_fun import generate_circ, get_dirname

def get_pairs(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr+1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs

def get_counter(subcircuit_circs, O_rho_pairs):
    counter = {}
    for subcircuit_idx, subcircuit_circ in enumerate(subcircuit_circs):
        counter[subcircuit_idx] = {'effective':subcircuit_circ.n_qubits,'rho':0,'O':0,'d':subcircuit_circ.n_qubits}
    for pair in O_rho_pairs:
        O_qubit, rho_qubit = pair
        counter[O_qubit[0]]['effective'] -= 1
        counter[O_qubit[0]]['O'] += 1
        counter[rho_qubit[0]]['rho'] += 1
    return counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--circuit_type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--circ_sizes', nargs='+', type=int,help='Benchmark circuit sizes')
    parser.add_argument('--cc_size', metavar='N',type=int,help='CC size')
    args = parser.parse_args()

    for full_circ_size in args.circ_sizes:
        dirname = get_dirname(circuit_type=args.circuit_type,cc_size=args.cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=None,field='generator')
        if os.path.exists(dirname):
            subprocess.run(['rm','-r',dirname])

        if args.circuit_type == 'adder':
            max_subcircuit_qubit = min(int(full_circ_size/1.5),args.cc_size)
        else:
            max_subcircuit_qubit = args.cc_size

        full_circuit = generate_circ(full_circ_size=full_circ_size,circuit_type=args.circuit_type)
        if full_circuit.n_qubits==0:
            continue
        else:
            solution_dict = find_cuts(circ=full_circuit, max_subcircuit_qubit=max_subcircuit_qubit)
        
        if solution_dict != {}:
            # solution_dict['model'].print_stat()
            os.makedirs(dirname)
            subcircuit_circs, complete_path_map, K, d = cut_circuit(circ=full_circuit,positions=solution_dict['positions'])
            num_instances = []
            num_collapsed = []
            for subcircuit_d, subcircuit_rho, subcircuit_O in zip(solution_dict['num_d_qubits'],solution_dict['num_rho_qubits'],solution_dict['num_O_qubits']):
                subcircuit_d = round(subcircuit_d)
                subcircuit_rho = round(subcircuit_rho)
                subcircuit_O = round(subcircuit_O)
                num_instances.append('%d-q * %d'%(subcircuit_d,4**subcircuit_rho*3**subcircuit_O))
                num_collapsed.append(subcircuit_d-subcircuit_O)
            result_str = colored(
                'MIP: {:d}-qubit full circuit {:d} cuts, subcircuits: {}, Cutter : {}'.format(full_circ_size,len(solution_dict['positions']),num_instances,d),
                'blue')
            print(result_str,flush=True)

            O_rho_pairs = get_pairs(complete_path_map=complete_path_map)
            counter = get_counter(subcircuit_circs=subcircuit_circs, O_rho_pairs=O_rho_pairs)
            case_dict = {}
            case_dict['full_circuit'] = full_circuit
            case_dict['subcircuit_circs'] = subcircuit_circs
            case_dict['complete_path_map'] = complete_path_map
            case_dict['counter'] = counter
            pickle.dump(case_dict, open('%s/subcircuits.pckl'%(dirname),'wb'))
        else:
            result_str = colored(
                'MIP: {:d}-qubit full circuit cannot be cut into {:d}-qubit subcircuits'.format(full_circ_size,args.cc_size),
                'red')
            print(result_str,flush=True)