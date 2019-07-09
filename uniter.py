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

def y_sigma_separator(y_sigma_freq, cregs, complete_path_map):
	print('separating y and sigma in the sub circuit')
	print('y_sigma_freq : ', y_sigma_freq)
	print('cregs : ', cregs)
	y_freq = {}
	sigma_products = {}
	for y_sigma_state in y_sigma_freq:
		print('looking at y_sigma_state: ', y_sigma_state)
		sigma_product = 1
		y_freq_state = ''
		for reg_idx, reg_state in enumerate(y_sigma_state.split(' ')):
			print('register ', cregs[reg_idx].name, ' has state ', reg_state)
			if 'measure' in cregs[reg_idx].name:
				for sigma_char in reg_state:
					sigma = -1 if int(sigma_char)==1 else 1
					# print('multiply sigma_product by ', sigma)
					sigma_product *= sigma
			elif 'output' in cregs[reg_idx].name:
				y_freq_state += reg_state
		print('y_freq_state = ', y_freq_state)
		print('sigma product = ', sigma_product)
		
		if y_freq_state != '' and y_freq_state not in y_freq:
			y_freq[y_freq_state] = y_sigma_freq[y_sigma_state]
		elif y_freq_state != '' and y_freq_state in y_freq:
			y_freq[y_freq_state] += y_sigma_freq[y_sigma_state]
		if y_freq_state != '' and y_freq_state not in sigma_products:
			sigma_products[y_freq_state] = sigma_product
		elif y_freq_state != '' and y_freq_state in sigma_products:
			sigma_products[y_freq_state] *= sigma_product
	print('y_freq for the subcircuit = ', y_freq)
	print('sigma_products for the subcircuit = ', sigma_products)
	return y_freq, sigma_products

def fragment_combiner(frag_0, frag_1, complete_path_map):
	# print('combine : ')
	# print(frag_0)
	# print(frag_1)
	combined_0_1 = {}

	frag_0_y_sigma_states = frag_0[0]
	frag_0_registers = frag_0[1]
	frag_1_y_sigma_states = frag_1[0]
	frag_1_registers = frag_1[1]

	for frag_0_state in frag_0_y_sigma_states:
		for frag_1_state in frag_1_y_sigma_states:
			if frag_0_state != '':
				key = frag_0_state + ' ' + frag_1_state
			else:
				key = frag_0_state + frag_1_state
			prob = frag_0_y_sigma_states[frag_0_state] * frag_1_y_sigma_states[frag_1_state]
			combined_0_1[key] = prob
	combined_registers = frag_0_registers + frag_1_registers
	combined_fragment = (combined_0_1, combined_registers)
	# print('combined_fragment = ', combined_fragment)
	return combined_fragment

def input_qubit_locator(complete_path_map, sub_circ_idx, qubit):
	for input_qubit in complete_path_map:
		path = complete_path_map[input_qubit]
		output_sub_circ_idx = path[len(path)-1][0]
		output_qubit = path[len(path)-1][2]
		if sub_circ_idx == output_sub_circ_idx and qubit == output_qubit:
			return input_qubit
	return None


def fragment_output_organizer(fragments, complete_path_map):
	# print('*' * 100)
	# print('calling fragment_output_organizer')

	t_s = {}

	empty_dict = {}
	empty_dict[''] = 1
	combined_fragments = (empty_dict, [])
	# print('base_frag = ', base_frag)
	for frag_to_combine in fragments:
		combined_fragments = fragment_combiner(combined_fragments, frag_to_combine, complete_path_map)

	# print('combined_fragment = ', combined_fragments)

	y_sigma_distribution = combined_fragments[0]
	y_sigma_registers = combined_fragments[1]

	y_state_length = 0
	for element in y_sigma_registers:
		if 'output' in element[1].name:
			y_state_length += element[1].size
	# print('y_state should have %d digits' % y_state_length)

	for y_sigma in y_sigma_distribution:
		prob = y_sigma_distribution[y_sigma]
		y_state = [-1 for x in range(y_state_length)]
		# print('y_sigma = ', y_sigma, 'prob = ', prob)
		# print('initialize y_state = ', y_state)
		for register_idx, register_measurement in enumerate(y_sigma.split(' ')):
			# print('looking at y_sigma', register_measurement)
			sub_circ_idx = y_sigma_registers[register_idx][0]
			register = y_sigma_registers[register_idx][1]
			if 'measure' in register.name:
				# print('is a measure, multiply sigma for ', register_measurement)
				for sigma in register_measurement:
					sigma = 1 if int(sigma) == 0 else -1
					prob *= sigma
			elif 'output' in register.name:
				# print('is an output')
				for register_offset, y in enumerate(register_measurement):
					input_qubit = input_qubit_locator(complete_path_map, sub_circ_idx, register[register_offset])
					# print('y register_measurement ', y, 'is for qubit:')
					# print(input_qubit)
					y_state[input_qubit[1]] = y
		final_y_state = ''
		for x in y_state:
			final_y_state += x
		# print('y_state = ', final_y_state)
		# print('prob after multiplying sigma = ', prob)
		if final_y_state in t_s:
			t_s[final_y_state] += prob
		else:
			t_s[final_y_state] = prob
	# print('t_s is: ', t_s)
	
	# print('finished fragment_output_organizer')
	# print('*' * 100)
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
		y_sigma_freq  = {}
		for key in counts:
			y_sigma_freq[key[::-1]] = counts[key]/num_shots
		# print('sub_circ_bridge %d y_sigma_freq distribution:' % sub_circ_idx, y_sigma_freq)
		# print('sub_circ_bridge %d cregs:'%sub_circ_idx, sub_circ.cregs)
		# sub_circ_output, y_reg_idx = y_sigma_separator(y_sigma_freq, sub_circ.cregs, complete_path_map)
		# print('sub_circ %d output:' % sub_circ_idx, sub_circ_output)
		print('sub_circ_bridge %d y_sigma measurement = ' % sub_circ_idx, y_sigma_freq)
		print('cregs : ', sub_circ.cregs)
		[(sub_circ_idx, creg) for creg in sub_circ.cregs]
		fragments.append((y_sigma_freq, [(sub_circ_idx, creg) for creg in sub_circ.cregs]))
	
	print('fragments for s = ', s, ' are:', fragments)
	t_s = fragment_output_organizer(fragments, complete_path_map)
	# print('combined output prob for sample: ', t_s)
	return t_s

def tensor_network_val_calc(sub_circs_no_bridge, complete_path_map, positions, num_samples, post_process_fn=None):
	cumulative_tn_val = {}
	all_s = sequential_s(len(positions))
	print('sequential_s = ', all_s)
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
            for i in range(1,9):
                all_s.append(prev_s + [i])
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
	dag_drawer(original_dag, filename='%s/original_dag.pdf' % path)
	q = circ.qregs[0]
	
	''' Test positions for the toy circuit'''
	# positions = [(q[4],1), (q[1],1)]
	positions = [(q[1],0), (q[0],0)]
	# positions = [(q[1],0)]

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