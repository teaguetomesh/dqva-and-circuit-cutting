from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.compiler import transpile
from qiskit.providers.aer import noise
import pickle
import itertools
import copy
import numpy as np
from time import time
import argparse
from utils.helper_fun import evaluate_circ, get_evaluator_info, get_circ_saturated_shots, distribute_cluster_shots, get_filename, read_file
from utils.conversions import dict_to_array, reverse_prob

def find_cluster_O_rho_qubits(complete_path_map,cluster_idx):
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for q in path[:-1]:
                if q[0] == cluster_idx:
                    O_qubits.append(q)
            for q in path[1:]:
                if q[0] == cluster_idx:
                    rho_qubits.append(q)
    return O_qubits, rho_qubits

def find_all_simulation_combinations(O_qubits, rho_qubits, num_qubits):
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','minus','plus_i','minus_i']
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
        complete_m = ['I' for i in range(num_qubits)]
        for i in range(len(meas)):
            complete_m[O_qubits[i][1]] = meas[i]
        complete_meas.append(complete_m)
    # print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    return combinations

def evaluate_cluster(complete_path_map, cluster_circ, combinations, backend, evaluator_info):
    cluster_prob = {}
    num_qubits = len(cluster_circ.qubits)
    uniform_p = 1/2**num_qubits
    uniform_prob = np.array([uniform_p for x in range(2**num_qubits)])
    for _, combination in enumerate(combinations):
        cluster_dag = circuit_to_dag(cluster_circ)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = cluster_circ.qubits[i]
            if x == 'zero':
                continue
            elif x == 'one':
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal initialization : ',x)
        for i,x in enumerate(meas):
            q = cluster_circ.qubits[i]
            if x == 'I':
                continue
            elif x == 'X':
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                cluster_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        cluster_circ_inst = dag_to_circuit(cluster_dag)
        cluster_prob[(tuple(inits),tuple(meas))] = uniform_prob
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
            complete_path_map = case_dict['complete_path_map']
            clusters = case_dict['clusters']
            case_dict['all_cluster_prob'] = {}
            for cluster_idx, cluster_circ in enumerate(clusters):
                O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
                combinations = find_all_simulation_combinations(O_qubits, rho_qubits, len(cluster_circ.qubits))
                print('Case {}, cluster_{:d} {:d}_qubits * {:d}_instances on fake QUANTUM SIMULATOR, '.format(case,cluster_idx,len(cluster_circ.qubits),len(combinations)))
                cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                        cluster_circ=cluster_circ,
                        combinations=combinations,
                        backend='fake',evaluator_info=None)
                case_dict['all_cluster_prob'][cluster_idx] = cluster_prob
            pickle.dump({case:case_dict}, open(output_filename,'ab'))
            counter += 1
            print('dumped case {}, {:d}/{:d} cases'.format(case,counter,len(evaluator_input)))
            print('-'*100)