import itertools
import numpy as np
from time import time
import argparse
import pickle
import os
import subprocess
from termcolor import colored

from utils.helper_fun import read_file, find_process_jobs, get_dirname

def get_combinations(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr+1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    basis = ['I+Z','I-Z','X','Y']
    combinations = list(itertools.product(basis,repeat=len(O_rho_pairs)))
    return O_rho_pairs, combinations

def find_init_meas(combination, O_rho_pairs, subcircuit_circs):
    # print('Finding init_meas for',combination)
    all_init_meas = {}
    for subcircuit_idx, subcircuit_circ in enumerate(subcircuit_circs):
        init = ['zero' for q in range(subcircuit_circ.n_qubits)]
        meas = ['comp' for q in range(subcircuit_circ.n_qubits)]
        all_init_meas[subcircuit_idx] = [init,meas]
    for s, pair in zip(combination, O_rho_pairs):
        O_qubit, rho_qubit = pair
        all_init_meas[rho_qubit[0]][0][rho_qubit[1]] = s
        all_init_meas[O_qubit[0]][1][O_qubit[1]] = s
    # print(all_init_meas)
    for subcircuit_idx in all_init_meas:
        init = all_init_meas[subcircuit_idx][0]
        init_combinations = []
        for idx, x in enumerate(init):
            if x == 'zero':
                init_combinations.append(['zero'])
            elif x == 'I+Z':
                init_combinations.append(['+zero'])
            elif x == 'I-Z':
                init_combinations.append(['+one'])
            elif x == 'X':
                init_combinations.append(['2plus','-zero','-one'])
            elif x == 'Y':
                init_combinations.append(['2plusI','-zero','-one'])
            else:
                raise Exception('Illegal initilization symbol :',x)
        init_combinations = list(itertools.product(*init_combinations))
        meas = all_init_meas[subcircuit_idx][1]
        meas_combinations = []
        for x in meas:
            if x == 'comp':
                meas_combinations.append(['comp'])
            elif x == 'I+Z':
                meas_combinations.append(['+I','+Z'])
            elif x == 'I-Z':
                meas_combinations.append(['+I','-Z'])
            elif x == 'X' or x == 'Y':
                meas_combinations.append(['+%s'%x])
            else:
                raise Exception('Illegal measurement symbol :',x)
        meas_combinations = list(itertools.product(*meas_combinations))
        subcircuit_init_meas = []
        for init in init_combinations:
            for meas in meas_combinations:
                subcircuit_init_meas.append((tuple(init),tuple(meas)))
        all_init_meas[subcircuit_idx] = subcircuit_init_meas
    # print(all_init_meas)
    return all_init_meas

def build(full_circuit, combinations, O_rho_pairs, subcircuit_circs, all_indexed_combinations, smart_order):
    kronecker_terms = {subcircuit_idx:{} for subcircuit_idx in range(len(subcircuit_circs))}
    summation_terms = []
    for i, combination in enumerate(combinations):
        # print('%d/%d combinations:'%(i+1,len(combinations)),combination)
        summation_term = []
        all_init_meas = find_init_meas(combination, O_rho_pairs, subcircuit_circs)
        for subcircuit_idx in smart_order:
            subcircuit_kron_term = []
            # print('Subcircuit_%d init_meas ='%subcircuit_idx,all_init_meas[subcircuit_idx])
            for init_meas in all_init_meas[subcircuit_idx]:
                # print('Subcircuit_%d init_meas ='%subcircuit_idx,init_meas)
                coefficient = 1
                init = list(init_meas[0])
                for idx, x in enumerate(init):
                    if x == 'zero':
                        continue
                    elif x == '+zero':
                        init[idx] = 'zero'
                    elif x == '+one':
                        init[idx] = 'one'
                    elif x == '2plus':
                        init[idx] = 'plus'
                        coefficient *= 2
                    elif x == '-zero':
                        init[idx] = 'zero'
                        coefficient *= -1
                    elif x == '-one':
                        init[idx] = 'one'
                        coefficient *= -1
                    elif x =='2plusI':
                        init[idx] = 'plusI'
                        coefficient *= 2
                    else:
                        raise Exception('Illegal initilization symbol :',x)
                meas = list(init_meas[1])
                for idx, x in enumerate(meas):
                    if x == 'comp':
                        continue
                    elif x == '+I':
                        meas[idx] = 'I'
                    elif x == '+Z':
                        meas[idx] = 'Z'
                    elif x == '-Z':
                        meas[idx] = 'Z'
                        coefficient *= -1
                    elif x =='+X':
                        meas[idx] = 'X'
                    elif x == '+Y':
                        meas[idx] = 'Y'
                    else:
                        raise Exception('Illegal measurement symbol :',x)
                init_meas = (tuple(init),tuple(meas))
                subcircuit_inst_index = all_indexed_combinations[subcircuit_idx][init_meas]
                subcircuit_kron_term.append((coefficient,subcircuit_inst_index))
                # print(coefficient,init_meas)
            subcircuit_kron_term = tuple(subcircuit_kron_term)
            # print('Subcircuit_%d kron term %d ='%(subcircuit_idx,subcircuit_inst_index),subcircuit_kron_term)
            if subcircuit_kron_term not in kronecker_terms[subcircuit_idx]:
                subcircuit_kron_index = len(kronecker_terms[subcircuit_idx])
                kronecker_terms[subcircuit_idx][subcircuit_kron_term] = subcircuit_kron_index
            else:
                subcircuit_kron_index = kronecker_terms[subcircuit_idx][subcircuit_kron_term]
            summation_term.append(subcircuit_kron_index)
        # print('Summation term =',summation_term,'\n')
        summation_terms.append(summation_term)
    # [print(subcircuit_idx,kronecker_terms[subcircuit_idx]) for subcircuit_idx in kronecker_terms]
    return kronecker_terms, summation_terms

def smart_subcircuit_order(counter, qubit_to_cut):
    smart_order = sorted(list(counter.keys()),key=lambda x:counter[x]['effective'])
    # print('Before blurry:',counter)
    # print('Before blurry:',smart_order)

    while qubit_to_cut > 0:
        for subcircuit_idx in smart_order[::-1]:
            num_qubits = counter[subcircuit_idx]['effective']
            if num_qubits > 0 and qubit_to_cut > 0:
                counter[subcircuit_idx]['effective'] = num_qubits - 1
                qubit_to_cut -= 1

    smart_order = sorted(list(counter.keys()),key=lambda x:counter[x]['effective'])
    # print('After blurry:',counter)
    # print('After blurry:',smart_order)
    return smart_order, counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--circuit_type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--circ_sizes', nargs='+', type=int,help='Benchmark circuit sizes')
    parser.add_argument('--cc_size', metavar='N',type=int,help='CC size')
    parser.add_argument('--techniques', nargs='+', type=int,help='Techniques : smart_order, early_termination, num_workers, qubit_limit')
    parser.add_argument('--eval_mode', type=str,help='Evaluation backend mode')
    args = parser.parse_args()

    _, _, num_workers, qubit_limit = args.techniques

    for full_circ_size in args.circ_sizes:
        source_folder = get_dirname(circuit_type=args.circuit_type,cc_size=args.cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=None,field='generator')
        eval_folder = get_dirname(circuit_type=args.circuit_type,cc_size=args.cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=args.eval_mode,field='evaluator')
        case_dict = read_file(filename='%s/subcircuits.pckl'%source_folder)
        all_indexed_combinations = read_file(filename='%s/all_indexed_combinations.pckl'%(eval_folder))
        if len(case_dict)==0:
            continue
        
        full_circuit = case_dict['full_circuit']
        subcircuit_circs = case_dict['subcircuit_circs']
        complete_path_map = case_dict['complete_path_map']
        counter = case_dict['counter']
        # [print(x,complete_path_map[x]) for x in complete_path_map]
        # [print(x.n_qubits) for x in subcircuit_circs]
        O_rho_pairs, combinations = get_combinations(complete_path_map=complete_path_map)
        qubit_to_cut = max(full_circ_size-qubit_limit,0)

        info_str = colored('Reconstructing %d-qubit full circuit : %d summation_terms'%(full_circ_size,len(combinations)),'blue')
        print(info_str,flush=True)

        smart_order = sorted(list(counter.keys()),key=lambda x:counter[x]['effective'])

        kronecker_terms, summation_terms = build(full_circuit=full_circuit, combinations=combinations,
        O_rho_pairs=O_rho_pairs, subcircuit_circs=subcircuit_circs, all_indexed_combinations=all_indexed_combinations,smart_order=smart_order)

        dest_folder = get_dirname(circuit_type=args.circuit_type,cc_size=args.cc_size,full_circ_size=full_circ_size,
        techniques=args.techniques,eval_mode=args.eval_mode,field='rank')
        if os.path.exists('%s'%(dest_folder)):
            subprocess.run(['rm','-r','%s'%(dest_folder)])
        os.makedirs('%s'%(dest_folder))
        for rank in range(num_workers):
            rank_summation_terms = find_process_jobs(jobs=summation_terms,rank=rank,num_workers=num_workers)
            # print('Rank %d has %d/%d summation terms'%(rank,len(rank_summation_terms),len(summation_terms)))
            rank_folder = '%s/rank_%d'%(dest_folder,rank)
            os.makedirs(rank_folder)
            
            summation_term_file = open('%s/summation_terms.txt'%(rank_folder),'w')
            summation_term_file.write('reconstruction_qubit %d num_subcircuits %d num_summation_terms %d num_cuts %d\n'%(
                min(full_circ_size,qubit_limit),len(smart_order),len(rank_summation_terms),len(O_rho_pairs)))
            for summation_term in rank_summation_terms:
                for subcircuit_idx,subcircuit_kron_index in zip(smart_order, summation_term):
                    summation_term_file.write('%d,%d '%(subcircuit_idx,subcircuit_kron_index))
                summation_term_file.write('\n')
            summation_term_file.close()
        
            subcircuit_kron_terms_file = open('%s/subcircuit_kron_terms.txt'%(rank_folder),'w')
            subcircuit_kron_terms_file.write('%d subcircuits\n'%len(kronecker_terms))
            for subcircuit_idx in kronecker_terms:
                rank_subcircuit_kron_terms = find_process_jobs(jobs=list(kronecker_terms[subcircuit_idx].keys()),rank=rank,num_workers=num_workers)
                subcircuit_kron_terms_file.write('subcircuit %d kron_terms %d num_effective %d\n'%(
                    subcircuit_idx,len(rank_subcircuit_kron_terms),counter[subcircuit_idx]['effective']))
                for subcircuit_kron_term in rank_subcircuit_kron_terms:
                    subcircuit_kron_terms_file.write('%d %d\n'%(kronecker_terms[subcircuit_idx][subcircuit_kron_term],len(subcircuit_kron_term)))
                    [subcircuit_kron_terms_file.write('%d,%d '%(x[0],x[1])) for x in subcircuit_kron_term]
                    subcircuit_kron_terms_file.write('\n')
                if rank==0:
                    print('Rank %d needs to vertical collapse %d/%d instances of subcircuit %d'%(rank,len(rank_subcircuit_kron_terms),len(kronecker_terms[subcircuit_idx]),subcircuit_idx),flush=True)
            subcircuit_kron_terms_file.close()
        
        # TODO: add 'collapsed' into counter after adding horizontal_collapse
        pickle.dump({'smart_order':smart_order,'counter':counter}, open('%s/meta_data.pckl'%(dest_folder),'wb'))