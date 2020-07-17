import itertools
import numpy as np
import pickle
import copy
from time import time, sleep
import argparse
import subprocess
import glob
from termcolor import colored

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from utils.helper_fun import read_file, evaluate_circ, find_process_jobs, get_dirname
from utils.schedule import DeviceScheduler, SimulatorScheduler

def find_subcircuit_O_rho_qubits(complete_path_map,subcircuit_idx):
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for q in path[:-1]:
                if q[0] == subcircuit_idx:
                    O_qubits.append(q)
            for q in path[1:]:
                if q[0] == subcircuit_idx:
                    rho_qubits.append(q)
    return O_qubits, rho_qubits

def find_all_combinations(O_qubits, rho_qubits, num_qubits):
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','plusI']
    # print('Rho qubits:',rho_qubits)
    all_inits = list(itertools.product(init_states,repeat=len(rho_qubits)))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(num_qubits)]
        for i in range(len(init)):
            complete_init[rho_qubits[i][1]] = init[i]
        complete_inits.append(complete_init)
    # print('initializations:',complete_inits)

    # print('O qubits:',O_qubits)
    all_meas = list(itertools.product(measurement_basis,repeat=len(O_qubits)))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['comp' for i in range(num_qubits)]
        for i in range(len(meas)):
            complete_m[O_qubits[i][1]] = meas[i]
        complete_meas.append(complete_m)
    # print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    
    indexed_combinations = {}
    ctr = 0
    for combination in combinations:
        inits, meas = combination
        mutated_meas = mutate_measurement_basis(meas)
        for meas in mutated_meas:
            indexed_combinations[(tuple(inits),tuple(meas))] = ctr
            ctr+=1
    return combinations, indexed_combinations

def mutate_measurement_basis(meas):
    if all(x!='I' for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != 'I':
                mutated_meas.append([x])
            else:
                mutated_meas.append(['I','Z'])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas

def get_subcircuit_inst(subcircuit_idx, subcircuit_circ, combinations):
    circ_dict = {}
    for combination_ctr, combination in enumerate(combinations):
        subcircuit_dag = circuit_to_dag(subcircuit_circ)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = subcircuit_circ.qubits[i]
            if x == 'zero':
                continue
            elif x == 'one':
                subcircuit_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus':
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus':
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plusI':
                subcircuit_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minusI':
                subcircuit_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal initialization : ',x)
        for i,x in enumerate(meas):
            q = subcircuit_circ.qubits[i]
            if x == 'I' or x == 'comp':
                continue
            elif x == 'X':
                subcircuit_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                subcircuit_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        subcircuit_circ_inst = dag_to_circuit(subcircuit_dag)
        circ_dict[(subcircuit_idx,tuple(inits),tuple(meas))] = {'circuit':subcircuit_circ_inst,'shots':max(8192,int(2**subcircuit_circ_inst.n_qubits))}
    return circ_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--circuit_type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--circ_sizes', nargs='+', type=int,help='Benchmark circuit sizes')
    parser.add_argument('--cc_size', metavar='N',type=int,help='CC size')
    parser.add_argument('--eval_workers', metavar='N',type=int,help='Total number of workers')
    parser.add_argument('--eval_rank', metavar='N',type=int,help='Rank of worker')
    parser.add_argument('--eval_mode', type=str,help='Evaluation backend mode')
    args = parser.parse_args()

    for full_circ_size in args.circ_sizes:
        source_folder = get_dirname(circuit_type=args.circuit_type,cc_size=args.cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=None,field='generator')
        case_dict = read_file(filename='%s/subcircuits.pckl'%source_folder)
        if len(case_dict)==0:
            continue
        
        eval_folder = get_dirname(circuit_type=args.circuit_type,cc_size=args.cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=args.eval_mode,field='evaluator')

        full_circuit = case_dict['full_circuit']
        subcircuit_circs = case_dict['subcircuit_circs']
        complete_path_map = case_dict['complete_path_map']
        counter = case_dict['counter']

        circ_dict = {}
        all_indexed_combinations = {}
        total_subcircuit_circ_inst = 0
        for subcircuit_idx, subcircuit_circ in enumerate(subcircuit_circs):
            O_qubits, rho_qubits = find_subcircuit_O_rho_qubits(complete_path_map=complete_path_map,subcircuit_idx=subcircuit_idx)
            total_subcircuit_circ_inst += 4**len(rho_qubits)*3**len(O_qubits)
            combinations, indexed_combinations = find_all_combinations(O_qubits, rho_qubits, subcircuit_circ.n_qubits)
            process_combinations = find_process_jobs(jobs=combinations,rank=args.eval_rank,num_workers=args.eval_workers)
            circ_dict.update(get_subcircuit_inst(subcircuit_idx=subcircuit_idx,subcircuit_circ=subcircuit_circ, combinations=process_combinations))
            all_indexed_combinations[subcircuit_idx] = indexed_combinations

        pickle.dump(all_indexed_combinations, open('%s/all_indexed_combinations.pckl'%(eval_folder),'wb'))
        
        scheduler = SimulatorScheduler(circ_dict=circ_dict,device_name='qasm')
        scheduler.run()
        scheduler.retrieve()
        circ_dict = scheduler.circ_dict

        # for key in circ_dict:
        #     subcircuit_idx, inits, meas = key
        #     subcircuit_inst_prob = circ_dict[key]['prob']
        #     mutated_meas = mutate_measurement_basis(meas)
        #     for meas in mutated_meas:
        #         index = all_indexed_combinations[subcircuit_idx][(tuple(inits),tuple(meas))]
        #         eval_file_name = '%s/%d_%d.txt'%(eval_folder,subcircuit_idx,index)
        #         eval_file = open(eval_file_name,'w')
        #         eval_file.write('%d %d\n'%(counter[subcircuit_idx]['d'],counter[subcircuit_idx]['effective']))
        #         [eval_file.write('%s '%x) for x in inits]
        #         eval_file.write('\n')
        #         [eval_file.write('%s '%x) for x in meas]
        #         eval_file.write('\n')
        #         [eval_file.write('%e '%x) for x in subcircuit_inst_prob]
        #         eval_file.close()