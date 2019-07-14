import argparse
from qiskit import QuantumCircuit
from qiskit import BasicAer, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.tools.visualization import dag_drawer
from cutting_help_fun import *
from qiskit.circuit import Measure
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
import networkx as nx
import os
import pickle
import random
import copy
from uniter import *

def toy_circ():
	num_qubits = 2
	q = QuantumRegister(num_qubits, 'q')
	circ = QuantumCircuit(q)
	circ.x(q[0])
	circ.x(q[0])
	circ.h(q[1])
	circ.x(q[1])
	return circ

def main():
    parser = argparse.ArgumentParser(description='Uniter for circuit cutting')
    parser.add_argument('--home-dir', type=str, default='.',help='home directory (default:.)')
    args = parser.parse_args()

    path = '%s/results' % args.home_dir
    if not os.path.isdir(path):
        os.makedirs(path)

    ''' Test example for a toy circuit'''
    circ = toy_circ()
    q = circ.qregs[0]
    positions = [(q[0],0)]

    ''' Test example for a supremacy circuit that will cut into 2 parts'''
    # circ = pickle.load(open('%s/supremacy_circuit_4_8.dump' % path, 'rb' ))
    # q = circ.qregs[0]
    # positions = [(q[2], 1), (q[7], 1), (q[10],1), (q[14], 1)]

    original_dag = circuit_to_dag(circ)

    cut_dag, path_order_dict = cut_edges(original_dag=original_dag, positions=positions)
    in_out_arg_dict = contains_wire_nodes(cut_dag)
    sub_reg_dicts, input_wires_mapping = sub_circ_reg_counter(cut_dag, in_out_arg_dict)

    K, d = cluster_character(sub_reg_dicts, positions)

    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    translation_dict = translation_dict_calc(input_wires_mapping, components, in_out_arg_dict, sub_reg_dicts)
    complete_path_map = complete_path_calc(path_order_dict, input_wires_mapping, translation_dict, sub_reg_dicts)

    sub_circs_no_bridge = generate_sub_circs(cut_dag, positions)

    c = ClassicalRegister(len(circ.qubits), 'c')
    meas = QuantumCircuit(q, c)
    meas.measure(q,c)
    qc = circ+meas
    backend_sim = BasicAer.get_backend('qasm_simulator')
    job_sim = execute(qc, backend_sim, shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    counts_rev = {}
    for key in counts:
        new_key = key[::-1]
        counts_rev[new_key] = counts[key] / 1024

    print('Measure original circuit:')
    print(qc)
    print('original circuit output prob = ', counts_rev)
    print('*' * 200)
    print('cut at positions : ', positions)
    print('cut_dag has %d connected components, K = %d, d = %d'
    % (nx.number_weakly_connected_components(cut_dag._multi_graph), K, d))

    for i, sub_circ_no_bridge in enumerate(sub_circs_no_bridge):
        print('sub circuit %d : ' % i)
        print(sub_circ_no_bridge)

    fragment_all_s, all_s = fragment_all_s_calc(sub_circs_no_bridge, complete_path_map, positions)
    fragment_all_s = coefficients_multiplier(fragment_all_s, positions, all_s, complete_path_map)
    fragment_all_s = combiner(fragment_all_s)
    reconstructed_prob = reconstructor(fragment_all_s, complete_path_map)

    print('reconstructed prob = ', reconstructed_prob)
    print('chi square distance = ', chiSquared(counts_rev, reconstructed_prob))
    print('*' * 200)

if __name__ == '__main__':
	main()