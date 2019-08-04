from qiskit import BasicAer, execute
import numpy as np

def simulate_circ(circ):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    outputprob = [np.power(abs(x),2) for x in outputstate]
    return outputprob

def generate_cluster_instances(cluster_circ, cluster_idx, complete_path_map):
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
    print('Modify output measurements for:')
    print(cut_edge_output_qubits)
    print('Modify input initialization for:')
    print(cut_edge_input_qubits)