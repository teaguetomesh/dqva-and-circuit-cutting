import argparse
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.tools.visualization import dag_drawer
from cutting_help_fun import *
from qiskit.circuit import Measure
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
import networkx as nx
import os

def toy_circ():
	num_qubits = 5
	q = QuantumRegister(num_qubits, 'q')
	circ = QuantumCircuit(q)
	circ.h(q)
	for i in range(num_qubits):
		circ.cx(q[i], q[(i+1)%num_qubits])
	return circ

def sub_circ_sampler(s, sub_cirs, complete_path_map):
	s_idx = 0
	for map_key in complete_path_map:
		path = complete_path_map[map_key]
		num_links = len(path) - 1
		for link_idx in range(num_links):
			sample_s = s[s_idx]
			s_idx += 1
			source_circ_dag = circuit_to_dag(sub_cirs[path[link_idx][0]])
			dest_circ_dag = circuit_to_dag(sub_cirs[path[link_idx+1][0]])
			dest_ancilla = path[link_idx+1][1]
			print('modify io for path:', path, 'sample_s = ', sample_s)
			if sample_s == 1:
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
			if sample_s == 2:
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_ancilla],
				cargs=[])
			if sample_s == 3:
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[path[link_idx][1]])
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_ancilla],
				cargs=[])
			if sample_s == 4:
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[path[link_idx][1]])
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_ancilla],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_ancilla],
				cargs=[])
			if sample_s == 5:
				source_circ_dag.apply_operation_back(op=SdgGate(), 
				qargs=[path[link_idx][1]])
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[path[link_idx][1]])
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
				dest_circ_dag.apply_operation_front(op=SGate(),
				qargs=[dest_ancilla],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_ancilla],
				cargs=[])
			if sample_s == 6:
				source_circ_dag.apply_operation_back(op=SdgGate(), 
				qargs=[path[link_idx][1]])
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[path[link_idx][1]])
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
				dest_circ_dag.apply_operation_front(op=SGate(),
				qargs=[dest_ancilla],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_ancilla],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_ancilla],
				cargs=[])
			if sample_s == 7:
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
			if sample_s == 8:
				source_circ_dag.apply_operation_back(op=Measure(), 
				qargs=[path[link_idx][1]],
				cargs=[path[link_idx][2]])
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_ancilla],
				cargs=[])
			sub_cirs[path[link_idx][0]] = dag_to_circuit(source_circ_dag)
			sub_cirs[path[link_idx+1][0]] = dag_to_circuit(dest_circ_dag)

	return sub_cirs

def main():
	parser = argparse.ArgumentParser(description='Uniter for circuit cutting')
	parser.add_argument('--home-dir', type=str, default='.',help='home directory (default:.)')
	args = parser.parse_args()

	path = '%s/results' % args.home_dir
	if not os.path.isdir(path):
		os.makedirs(path)

	circ = toy_circ()

	original_dag = circuit_to_dag(circ)
	dag_drawer(original_dag, filename='%s/original_dag.pdf' % path)
	q = circ.qregs[0]
	
	''' Test positions that will cut into 3 parts'''
	positions = [(q[1],1), (q[0],1)]

	cut_dag, path_order_dict = cut_edges(original_dag=original_dag, positions=positions)
	dag_drawer(cut_dag, filename='%s/cut_dag.pdf' % path)
	in_out_arg_dict = contains_wire_nodes(cut_dag)
	sub_reg_dicts, input_wires_mapping = sub_circ_reg_counter(cut_dag, in_out_arg_dict)

	K, d = cluster_character(sub_reg_dicts, positions)
	print('cut_dag has %d connected components, K = %d, d = %d'
	% (nx.number_weakly_connected_components(cut_dag._multi_graph), K, d))

	print('\ninput_wires_mapping:')
	[print(x, input_wires_mapping[x]) for x in input_wires_mapping]

	components = list(nx.weakly_connected_components(cut_dag._multi_graph))
	translation_dict = translation_dict_calc(input_wires_mapping, components, in_out_arg_dict, sub_reg_dicts)
	complete_path_map = complete_path_calc(path_order_dict, input_wires_mapping, translation_dict, sub_reg_dicts)

	print('\ntranslation_dict:')
	[print(x, translation_dict[x]) for x in translation_dict]
	print('\ncomplete_path_map:')
	[print(x, complete_path_map[x]) for x in complete_path_map]

	sub_circs = generate_sub_circs(cut_dag, positions)
	for idx, sub_circ in enumerate(sub_circs):
		sub_circ.draw(output='text',line_length = 400, filename='%s/sub_circ_%d.txt' % (path, idx))
		dag_drawer(circuit_to_dag(sub_circ), filename='%s/sub_dag_%d.pdf' % (path, idx))

	s = [7,8]
	sub_circs_sample = sub_circ_sampler(s, sub_circs, complete_path_map)
	for idx, sub_circ in enumerate(sub_circs):
		sub_circ.draw(output='text',line_length = 400, filename='%s/25_sub_circ_%d.txt' % (path, idx))
		dag_drawer(circuit_to_dag(sub_circ), filename='%s/25_sub_dag_%d.pdf' % (path, idx))

if __name__ == '__main__':
	main()