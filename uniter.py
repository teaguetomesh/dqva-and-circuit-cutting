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

def toy_circ():
	num_qubits = 3
	q = QuantumRegister(num_qubits, 'q')
	circ = QuantumCircuit(q)
	# circ.h(q)
	# for i in range(num_qubits):
	# 	circ.cx(q[i], q[(i+1)%num_qubits])
	circ.h(q[0])
	circ.cx(q[1],q[2])
	circ.cx(q[1],q[0])
	return circ

def sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map):
	# print('building bridges for s = ', s)
	s_idx = 0
	# sub_circs_bridge = [x for x in sub_circs_no_bridge]
	sub_circs_bridge = copy.deepcopy(sub_circs_no_bridge)

	for map_key in complete_path_map:
		path = complete_path_map[map_key]
		num_links = len(path) - 1
		for link_idx in range(num_links):
			# print('adding bridge for path:', path[link_idx])
			sample_s = s[s_idx]
			s_idx += 1
			source_circ_dag = circuit_to_dag(sub_circs_bridge[path[link_idx][0]])
			dest_circ_dag = circuit_to_dag(sub_circs_bridge[path[link_idx+1][0]])
			dest_ancilla = path[link_idx+1][1]
			# print('modify io for path:', path, 'sample_s = ', sample_s)
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
				# print('modifying for sample_s', sample_s, 'qargs = ', [path[link_idx][1]])
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
			sub_circs_bridge[path[link_idx][0]] = dag_to_circuit(source_circ_dag)
			sub_circs_bridge[path[link_idx+1][0]] = dag_to_circuit(dest_circ_dag)
		output_qubit_sub_circ = path[len(path)-1][0]
		output_qubit = path[len(path)-1][1]
		output_qubit_measure = path[len(path)-1][2]
		# print(map_key, ' output is sub_circ', output_qubit_sub_circ, output_qubit, output_qubit_measure)
		sub_circs_bridge[output_qubit_sub_circ].append(instruction=Measure(),qargs=[output_qubit],cargs=[output_qubit_measure])

	return sub_circs_bridge

def y_sigma_separator(counts, cregs):
	# print('separating y and sigma in the sub circuit')
	sigma_reg_idx = None
	y_reg_idx = None
	for creg_idx, creg in enumerate(cregs):
		if 'measure' in creg.name:
			sigma_reg_idx = creg_idx
			break
	for creg_idx, creg in enumerate(cregs):
		if 'output' in creg.name:
			y_reg_idx = creg_idx
			break
	# print('sigma register: ', sigma_reg_idx, ' y register: ', y_reg_idx)
	new_counts = {}
	for key in counts:
		sigma_y = key.split(' ')
		# print('sigma_y = ', sigma_y)
		sigma_product = 1
		if sigma_reg_idx != None:
			sigmas = sigma_y[len(sigma_y) - 1 - sigma_reg_idx]
			for sigma_char in sigmas:
				sigma = -1 if int(sigma_char)==1 else 1
				sigma_product *= sigma

		ys = sigma_y[len(sigma_y) - 1 - y_reg_idx][::-1]
		# print('sigmas = ', sigmas, 'sigma_product = ', sigma_product, 'y = ', ys)
		# print('original count = ', counts[key], 'modified count = ', counts[key]*sigma_product)
		new_counts_key = ys
		new_counts_val = counts[key]*sigma_product
		if new_counts_key in new_counts:
			new_counts[new_counts_key] += new_counts_val
		else:
			new_counts[new_counts_key] = new_counts_val
	# print('new counts = ', new_counts)
	return new_counts, y_reg_idx

def fragment_output_organizer(fragments, complete_path_map):
	fragments_output_orders = {}
	for fragment_idx, fragment in enumerate(fragments):
		fragment_output_position = []
		# print(fragment)
		sub_circ_idx, y_reg, _ = fragment
		for i in range(y_reg.size):
			# print('looking for:', y_reg[i], 'in sub_circ ', sub_circ_idx)
			for input_qubit in complete_path_map:
				path = complete_path_map[input_qubit]
				output_sub_circ_idx = path[len(path)-1][0]
				output_cl_bit = path[len(path)-1][2]
				if output_sub_circ_idx == sub_circ_idx and y_reg[i] == output_cl_bit:
					# print('input qubit is:', input_qubit)
					input_position = list(complete_path_map.keys()).index(input_qubit)
					fragment_output_position.append(input_position)
		fragments_output_orders[fragment_idx] = fragment_output_position
	print('fragments_output_orders', fragments_output_orders)
	# print('fragments:', fragments)
	t_s = {}
	for final_measurement_output in range(np.power(2, len(complete_path_map))):
		'''Loop over all 2^n final output states'''
		final_measurement_output_binary = bin(final_measurement_output)[2:].zfill(len(complete_path_map))
		# print('final measurement output state = ', final_measurement_output_binary)

		final_measurement_output_prob = 1
		for fragment_idx, fragment in enumerate(fragments):
			_, _, sub_circ_output = fragments[fragment_idx]
			qubit_output_positions = fragments_output_orders[fragment_idx]
			sub_circ_output_key = ''
			for position in qubit_output_positions:
				sub_circ_output_key += final_measurement_output_binary[position]
			# print('fragment ', fragment_idx, 'key = ', type(sub_circ_output_key), sub_circ_output_key)
			if sub_circ_output_key in sub_circ_output:
				# print('multiply ', sub_circ_output[sub_circ_output_key])
				final_measurement_output_prob *= sub_circ_output[sub_circ_output_key]
			else:
				# print('multiply 0')
				final_measurement_output_prob *= 0
		# print('prob = ', final_measurement_output_prob)
		t_s[final_measurement_output_binary] = final_measurement_output_prob	
	return t_s

def ts_sampler(s, sub_circs_no_bridge, complete_path_map, num_shots=1024, post_process_fn=None):
	sub_circs_bridge = sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map)
	backend_sim = BasicAer.get_backend('qasm_simulator')
	fragments = []
	for sub_circ_idx, sub_circ in enumerate(sub_circs_bridge):
		print('sub_circ_bridge %d' % (sub_circ_idx))
		print(sub_circ)
		job_sim = execute(sub_circ, backend_sim, shots=num_shots)
		result_sim = job_sim.result()
		# print(type(result_sim), result_sim)
		counts = result_sim.get_counts(sub_circ)
		print('sub_circ_bridge %d raw counts:' % sub_circ_idx, counts)
		# print('sub_circ_bridge %d cregs:'%sub_circ_idx, sub_circ.cregs)
		sub_circ_output, y_reg_idx = y_sigma_separator(counts, sub_circ.cregs)
		# print('sub_circ %d output:' % sub_circ_idx, sub_circ_output)
		for x in sub_circ_output:
			sub_circ_output[x] /= num_shots
		print('sub_circ_bridge %d output measure: ' % sub_circ_idx, sub_circ_output)
		fragment = (sub_circ_idx, sub_circ.cregs[y_reg_idx], sub_circ_output)
		# print('fragment:', fragment)
		fragments.append(fragment)
	t_s = fragment_output_organizer(fragments, complete_path_map)
	# print('combined output prob for sample: ', t_s)
	return t_s

def tensor_network_val_calc(sub_circs_no_bridge, complete_path_map, positions, num_samples, post_process_fn=None):
	cumulative_tn_val = {}
	all_s = sequential_s(len(positions))
	print('sequential_s = ', all_s, 'len(positions) = ', len(positions))
	num_trials = 0
	# for i in range(num_samples):
	# 	s = random_s(len(positions))
	for i in range(np.power(8, len(positions))):
		s = all_s[i]
		print('*' * 200)
		print('sampling for s = ', s)
		c_s = 1
		for link_s in s:
			if link_s == 4 or link_s == 6 or link_s == 8:
				c_s *= -0.5
			else:
				c_s *= 0.5
		print('c_s = ', c_s)
		t_s = ts_sampler(s, sub_circs_no_bridge, complete_path_map)
		print('t_s = ', t_s)
		single_shot_tn_val = {}
		for output_state in t_s:
			single_shot_tn_val[output_state] = np.power(8,len(positions)) * c_s * t_s[output_state]
			# single_shot_tn_val[output_state] = c_s * t_s[output_state]
		print('single shot tensor network value = ', single_shot_tn_val)
		print('*' * 200)
		for key in single_shot_tn_val:
			if key in cumulative_tn_val:
				cumulative_tn_val[key] += single_shot_tn_val[key]
			else:
				cumulative_tn_val[key] = single_shot_tn_val[key]
		num_trials += 1
	for output_state in cumulative_tn_val:
		cumulative_tn_val[output_state] /= num_trials
	return cumulative_tn_val

def sequential_s(length):
	if length == 1:
		return [[1],[2],[3],[4],[5],[6],[7],[8]]
	else:
		all_prev_s = sequential_s(length - 1)
		all_s = []
		for prev_s in all_prev_s:
			for i in range(1,8):
				prev_s.append(i)
				all_s.append(prev_s)
	return all_s

def random_s(length):
	s = [random.randint(1,8) for i in range(length)]
	return s

def main():
	parser = argparse.ArgumentParser(description='Uniter for circuit cutting')
	parser.add_argument('--home-dir', type=str, default='.',help='home directory (default:.)')
	args = parser.parse_args()

	path = '%s/results' % args.home_dir
	if not os.path.isdir(path):
		os.makedirs(path)

	circ = toy_circ()
	# circ.draw(output='text',line_length = 400, filename='%s/original_circ.txt' % path)
	print('Original circuit:')
	print(circ)
	# circ = pickle.load(open('%s/supremacy_circuit_4_8.dump' % path, 'rb' ))

	original_dag = circuit_to_dag(circ)
	# dag_drawer(original_dag, filename='%s/original_dag.pdf' % path)
	q = circ.qregs[0]
	
	''' Test positions for the toy circuit'''
	# positions = [(q[4],1), (q[1],1)]
	positions = [(q[1],0)]

	''' Test positions that will cut into 2 parts'''
	# positions = [(q[2], 1), (q[7], 1), (q[10],1), (q[14], 1)]

	cut_dag, path_order_dict = cut_edges(original_dag=original_dag, positions=positions)
	# dag_drawer(cut_dag, filename='%s/cut_dag.pdf' % path)
	in_out_arg_dict = contains_wire_nodes(cut_dag)
	sub_reg_dicts, input_wires_mapping = sub_circ_reg_counter(cut_dag, in_out_arg_dict)

	# print('sub_reg_dicts:')
	# for reg_dict in sub_reg_dicts:
	# 	print(reg_dict)

	K, d = cluster_character(sub_reg_dicts, positions)
	print('\ncut_dag has %d connected components, K = %d, d = %d'
	% (nx.number_weakly_connected_components(cut_dag._multi_graph), K, d))

	# print('\ninput_wires_mapping:')
	# [print(x, input_wires_mapping[x]) for x in input_wires_mapping]

	components = list(nx.weakly_connected_components(cut_dag._multi_graph))
	translation_dict = translation_dict_calc(input_wires_mapping, components, in_out_arg_dict, sub_reg_dicts)
	complete_path_map = complete_path_calc(path_order_dict, input_wires_mapping, translation_dict, sub_reg_dicts)

	# print('\ntranslation_dict:')
	# [print(x, translation_dict[x]) for x in translation_dict]
	print('\ncomplete_path_map:')
	[print(x, complete_path_map[x]) for x in complete_path_map]

	sub_circs_no_bridge = generate_sub_circs(cut_dag, positions)
	for idx, sub_circ in enumerate(sub_circs_no_bridge):
		sub_circ.draw(output='text',line_length = 400, filename='%s/sub_circ_no_bridge_%d.txt' % (path, idx))
		print('sub_circ_no_bridge %d' % idx)
		print(sub_circ)
		# dag_drawer(circuit_to_dag(sub_circ), filename='%s/sub_dag_%d.pdf' % (path, idx))
	
	# s = random_s(len(positions))
	# sub_circs_bridge = sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map)
	# for idx, sub_circ in enumerate(sub_circs_bridge):
	# 	print('sub_circ_bridge %d' % idx)
	# 	print(sub_circ)
	c = ClassicalRegister(3, 'c')
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
	
	result = tensor_network_val_calc(sub_circs_no_bridge, complete_path_map, positions, 10, post_process_fn=None)
	print('Measure original circuit:')
	print(qc)
	print('original circ output prob = ', counts_rev)
	cut_circ_total_prob = 0
	for x in result.values():
		cut_circ_total_prob += x
	print('with cutting:', result)
	print('sum of probabilities = ', cut_circ_total_prob)

if __name__ == '__main__':
	main()