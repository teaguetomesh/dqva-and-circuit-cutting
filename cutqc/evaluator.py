import itertools, copy
import numpy as np
from time import time
from termcolor import colored
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate

from qiskit_helper_functions.non_ibmq_functions import read_dict, find_process_jobs, evaluate_circ

def find_subcircuit_O_rho_qubits(complete_path_map,subcircuit_idx):
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for q in path[:-1]:
                if q['subcircuit_idx'] == subcircuit_idx:
                    O_qubits.append(q)
            for q in path[1:]:
                if q['subcircuit_idx'] == subcircuit_idx:
                    rho_qubits.append(q)
    return O_qubits, rho_qubits

def find_all_combinations(O_qubits, rho_qubits, qubits):
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','plusI']
    # print('\u03C1 qubits :',rho_qubits)
    all_inits = list(itertools.product(init_states,repeat=len(rho_qubits)))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(len(qubits))]
        for i in range(len(init)):
            complete_init[qubits.index(rho_qubits[i]['subcircuit_qubit'])] = init[i]
        complete_inits.append(complete_init)
    # print('initializations:',complete_inits)

    # print('O qubits:',O_qubits)
    all_meas = list(itertools.product(measurement_basis,repeat=len(O_qubits)))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['comp' for i in range(len(qubits))]
        for i in range(len(meas)):
            complete_m[qubits.index(O_qubits[i]['subcircuit_qubit'])] = meas[i]
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

def get_subcircuit_instance(subcircuit_idx, subcircuit, combinations):
    circ_dict = {}
    for combination_ctr, combination in enumerate(combinations):
        # print('combination %d :'%combination_ctr,combination)
        subcircuit_dag = circuit_to_dag(subcircuit)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = subcircuit.qubits[i]
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
            q = subcircuit.qubits[i]
            if x == 'I' or x == 'comp':
                continue
            elif x == 'X':
                subcircuit_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                subcircuit_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        subcircuit_inst = dag_to_circuit(subcircuit_dag)
        # NOTE: Adjust subcircuit shots here
        num_shots = max(8192,int(2**subcircuit_inst.num_qubits))
        num_shots = min(8192*10,num_shots)
        circ_dict[(subcircuit_idx,tuple(inits),tuple(meas))] = {'circuit':subcircuit_inst,'shots':num_shots}
    return circ_dict

def sv_simulate(key,circuit,eval_folder,counter):
    all_indexed_combinations = read_dict('%s/all_indexed_combinations.pckl'%(eval_folder))
    subcircuit_idx, inits, meas = key
    subcircuit_inst_prob = evaluate_circ(circuit=circuit,backend='statevector_simulator')
    mutated_meas = mutate_measurement_basis(meas)
    for meas in mutated_meas:
        index = all_indexed_combinations[subcircuit_idx][(tuple(inits),tuple(meas))]
        eval_file_name = '%s/raw_%d_%d.txt'%(eval_folder,subcircuit_idx,index)
        # print('running',eval_file_name)
        eval_file = open(eval_file_name,'w')
        eval_file.write('d=%d effective=%d\n'%(counter[subcircuit_idx]['d'],counter[subcircuit_idx]['effective']))
        [eval_file.write('%s '%x) for x in inits]
        eval_file.write('\n')
        [eval_file.write('%s '%x) for x in meas]
        eval_file.write('\n')
        [eval_file.write('%e '%x) for x in subcircuit_inst_prob]
        eval_file.close()

def runtime_simulate(key,circuit,eval_folder,counter):
    all_indexed_combinations = read_dict('%s/all_indexed_combinations.pckl'%(eval_folder))
    subcircuit_idx, inits, meas = key
    uniform_p = 1/2**circuit.num_qubits
    subcircuit_inst_prob = [uniform_p] * int(2**circuit.num_qubits)
    mutated_meas = mutate_measurement_basis(meas)
    for meas in mutated_meas:
        index = all_indexed_combinations[subcircuit_idx][(tuple(inits),tuple(meas))]
        eval_file_name = '%s/raw_%d_%d.txt'%(eval_folder,subcircuit_idx,index)
        # print('running',eval_file_name)
        eval_file = open(eval_file_name,'w')
        eval_file.write('d=%d effective=%d\n'%(counter[subcircuit_idx]['d'],counter[subcircuit_idx]['effective']))
        [eval_file.write('%s '%x) for x in inits]
        eval_file.write('\n')
        [eval_file.write('%s '%x) for x in meas]
        eval_file.write('\n')
        [eval_file.write('%e '%x) for x in subcircuit_inst_prob]
        eval_file.close()