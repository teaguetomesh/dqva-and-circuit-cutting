from qiskit import BasicAer, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.tools.visualization import dag_drawer
import numpy as np
import itertools
import copy
import timeit

def simulate_circ(circ, simulator='statevector_simulator'):
    backend = BasicAer.get_backend(simulator)
    job = execute(circ, backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    outputprob = [np.power(abs(x),2) for x in outputstate]
    return outputprob

def generate_cluster_instances(cluster_circ, cluster_idx, complete_path_map):
    cluster_instances_outputprob = {}
    cut_edge_output_qubits = []
    cut_edge_input_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for idx, (circ_idx, path_qubit) in enumerate(path):
                if idx == 0 and circ_idx==cluster_idx:
                    # print('Only modify measurement output for qubit', path_qubit)
                    cut_edge_output_qubits.append(path_qubit)
                elif idx == len(path)-1 and circ_idx==cluster_idx:
                    # print('Only modify initialization for qubit', path_qubit)
                    cut_edge_input_qubits.append(path_qubit)
                elif circ_idx==cluster_idx:
                    # print('Modify both initialization and measurement for qubit', path_qubit)
                    cut_edge_output_qubits.append(path_qubit)
                    cut_edge_input_qubits.append(path_qubit)
    print('cluster circ %d'%cluster_idx)
    # print('Modify output measurements for:')
    # print(cut_edge_output_qubits)
    # print('Modify input initialization for:')
    # print(cut_edge_input_qubits)
    meas_modifications = len(cut_edge_output_qubits)
    perms = list(itertools.product(range(1,7),repeat=len(cut_edge_input_qubits)+len(cut_edge_output_qubits)))
    start = timeit.default_timer()
    for s_idx, s in enumerate(perms):
        cluster_circ_copy = copy.deepcopy(cluster_circ)
        cluster_dag = circuit_to_dag(cluster_circ_copy)
        # print(s)
        for idx, cut_s in enumerate(s[:meas_modifications]):
            qubit = cut_edge_output_qubits[idx]
            # print('modify measurement for qubit',qubit,'with s =', cut_s)
            if cut_s == 1 or cut_s == 2 or cut_s == 7 or cut_s == 8:
                continue
            if cut_s == 3 or cut_s == 4:
                cluster_dag.apply_operation_back(op=HGate(),qargs=[qubit])
            if cut_s == 5 or cut_s == 6:
                cluster_dag.apply_operation_back(op=SdgGate(),qargs=[qubit])
                cluster_dag.apply_operation_back(op=HGate(),qargs=[qubit])
        for idx, cut_s in enumerate(s[meas_modifications:]):
            qubit = cut_edge_input_qubits[idx]
            # print('modify initialization for qubit',qubit,'with s =', cut_s)
            if cut_s == 1 or cut_s == 7:
                continue
            if cut_s == 2 or cut_s == 8:
                cluster_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
            if cut_s == 3:
                cluster_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
            if cut_s == 4:
                cluster_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
            if cut_s == 5:
                cluster_dag.apply_operation_front(op=SGate(),qargs=[qubit],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
            if cut_s == 6:
                cluster_dag.apply_operation_front(op=SGate(),qargs=[qubit],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
        cluster_instances_outputprob[s] = simulate_circ(dag_to_circuit(cluster_dag))
        if s_idx%100 == 99:
            stop = timeit.default_timer()
            time_remaining = (stop-start)/((s_idx+1)/len(perms))-(stop-start)
            print('%d/%d = %.2f %% completed, estimated time remaining = %.2f seconds' %
            (s_idx+1,len(perms), (100*(s_idx+1)/len(perms)),time_remaining))
    stop = timeit.default_timer()
    print('Total runtime for cluster %d is %.2e seconds' % (cluster_idx, stop-start))
    print('*'*100)
    return cluster_instances_outputprob

def simulate(clusters, complete_path_map):
    simulator_output = {}
    for cluster_idx, cluster in enumerate(clusters):
        cluster_instances_outputprob = generate_cluster_instances(cluster,cluster_idx,complete_path_map)
        simulator_output[cluster_idx] = cluster_instances_outputprob