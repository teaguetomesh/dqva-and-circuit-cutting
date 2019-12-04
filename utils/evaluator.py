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
import os

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

def evaluate_cluster(complete_path_map, cluster_idx, cluster_circ):
    O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map=complete_path_map,cluster_idx=cluster_idx)
    combinations = find_all_simulation_combinations(O_qubits=O_qubits, rho_qubits=rho_qubits, num_qubits=len(cluster_circ.qubits))
    cluster_instances = {}
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
        cluster_instances[(tuple(inits),tuple(meas))] = cluster_circ_inst
    return cluster_instances

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator')
    parser.add_argument('--experiment-name', metavar='S', type=str,help='which experiment to reconstruct')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device input file to run')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    dirname, evaluator_input_filename = get_filename(experiment_name=args.experiment_name,
    circuit_type=args.circuit_type,
    device_name=args.device_name,
    evaluation_method=None,field='evaluator_input')
    evaluator_input_filename = dirname+evaluator_input_filename

    dirname, job_submittor_input_filename = get_filename(experiment_name=args.experiment_name,
    circuit_type=args.circuit_type,
    device_name=args.device_name,
    evaluation_method=None,field='job_submittor_input')
    job_submittor_input_filename = dirname+job_submittor_input_filename
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print('-'*50,'Generate Cluster Circuits','-'*50,flush=True)
    evaluator_input = read_file(filename=evaluator_input_filename)

    for case in evaluator_input:
        print('case {}'.format(case))
        case_dict = copy.deepcopy(evaluator_input[case])
        case_dict['all_cluster_prob'] = {}
        clusters = case_dict['clusters']
        complete_path_map = case_dict['complete_path_map']
        for cluster_idx, cluster_circ in enumerate(clusters):
            cluster_instances = evaluate_cluster(complete_path_map=complete_path_map,
            cluster_idx=cluster_idx,cluster_circ=cluster_circ)
            case_dict['all_cluster_prob'][cluster_idx] = cluster_instances
            print('cluster %d has %d instances'%(cluster_idx,len(cluster_instances)))
        pickle.dump({case:case_dict},open(job_submittor_input_filename,'ab'))