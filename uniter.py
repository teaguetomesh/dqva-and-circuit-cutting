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
from colorama import Fore, Back, Style

def toy_circ():
	num_qubits = 2
	q = QuantumRegister(num_qubits, 'q')
	circ = QuantumCircuit(q)
	# circ.h(q)
	# for i in range(num_qubits):
	# 	circ.cx(q[i], q[(i+1)%num_qubits])
	circ.x(q[0])
	circ.x(q[0])
	circ.h(q[1])
	circ.x(q[1])
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


def fragment_output_organizer(fragments, complete_path_map):
	print('*' * 100)
	print('calling fragment_output_organizer')

	t_s = {}

	empty_dict = {}
	empty_dict[''] = 1
	combined_fragments = (empty_dict, [])
	for frag_to_combine in fragments:
		combined_fragments = fragment_combiner(combined_fragments, frag_to_combine, complete_path_map)

	print('combined_fragment = ', combined_fragments)

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
		print('y_sigma = ', y_sigma, 'prob = ', prob)
		print('initialize y_state = ', y_state)
		for register_idx, register_measurement in enumerate(y_sigma.split(' ')):
			print('looking at y_sigma', register_measurement)
			sub_circ_idx = y_sigma_registers[register_idx][0]
			register = y_sigma_registers[register_idx][1]
			if 'measure' in register.name:
				print('is a measure, multiply sigma for ', register_measurement)
				for sigma in register_measurement:
					sigma = 1 if int(sigma) == 0 else -1
					prob *= sigma
			elif 'output' in register.name:
				print('is an output')
				for register_offset, y in enumerate(register_measurement):
					input_qubit = input_qubit_locator(complete_path_map, sub_circ_idx, register[register_offset])
					print('y register_measurement ', y, 'is for qubit:')
					print(input_qubit)
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
	
	print('finished fragment_output_organizer')
	print('*' * 100)
	return t_s

def fragment_s_calc(s, sub_circs_no_bridge, complete_path_map, num_shots=1024):
	# print('*' * 100)
	# print('calling ts_sampler')
	sub_circs_bridge = sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map)
	backend_sim = BasicAer.get_backend('qasm_simulator')
	fragment_s = []

	for sub_circ_idx, sub_circ in enumerate(sub_circs_bridge):
		print('sub_circ_bridge %d' % (sub_circ_idx))
		print(sub_circ)
		job_sim = execute(sub_circ, backend_sim, shots=num_shots)
		result_sim = job_sim.result()
		# print(type(result_sim), result_sim)
		counts = result_sim.get_counts(sub_circ)
		y_sigma_freq  = {}
		for key in counts:
			# y_sigma_freq[key] = counts[key]/num_shots
			y_sigma_freq[key[::-1]] = counts[key]/num_shots
		# print('sub_circ_bridge %d y_sigma_freq distribution:' % sub_circ_idx, y_sigma_freq)
		# print('sub_circ_bridge %d cregs:'%sub_circ_idx, sub_circ.cregs)
		# sub_circ_output, y_reg_idx = y_sigma_separator(y_sigma_freq, sub_circ.cregs, complete_path_map)
		# print('sub_circ %d output:' % sub_circ_idx, sub_circ_output)
		print('sub_circ_bridge %d y_sigma measurement = ' % sub_circ_idx, y_sigma_freq)
		print('cregs : ', sub_circ.cregs)
		[(sub_circ_idx, creg) for creg in sub_circ.cregs]
		fragment_s.append((y_sigma_freq, [(sub_circ_idx, creg) for creg in sub_circ.cregs]))
	
	print('fragments for s = ', s, ' are:', fragment_s)
	# t_s = fragment_output_organizer(fragments, complete_path_map)
	# print('combined output prob for sample: ', t_s)
	# print('finished ts_sampler')
	# print('*' * 100)
	return fragment_s

def fragment_all_s_calc(sub_circs_no_bridge, complete_path_map, positions):
	print('*' * 200)
	print(Fore.RED + 'calling fragment_all_s' + Style.RESET_ALL)
	all_s = sequential_s(len(positions))
	print('sequential_s = ', all_s)
	fragment_all_s = {}
	# for i in range(np.power(8, len(positions))):
	for s in all_s:
		key = ''
		for char in s:
			key += str(char)
		# s = all_s[i]
		fragment_s = fragment_s_calc(s, sub_circs_no_bridge, complete_path_map)
		fragment_all_s[key] = fragment_s
		# fragment_all_s.append((fragment_s,s))
	print(Fore.RED + 'fragment_all_s returns : ' + Style.RESET_ALL)
	[print(x, fragment_all_s[x]) for x in fragment_all_s]
	print('*' * 200)
	return fragment_all_s, all_s

def link_fragment_idx_calc(complete_path_map, positions, cut_idx):
	cut_qubit, cut_gate_idx = positions[cut_idx]
	# print('cut on qubit : ', cut_qubit, ' after gate ', cut_gate_idx)
	path = complete_path_map[cut_qubit]
	# print('path = ', path)
	link_start_idx = 0
	for position in positions:
		qubit, gate_idx = position
		if qubit == cut_qubit and gate_idx < cut_gate_idx:
			link_start_idx += 1
	start_frag_idx = path[link_start_idx][0]
	end_frag_idx = path[link_start_idx+1][0]
	return start_frag_idx, end_frag_idx

def coefficients_multiplier(fragment_all_s, positions, all_s, complete_path_map):
	for cut_idx, cut in enumerate(positions):
		start_frag_idx, end_frag_idx = link_fragment_idx_calc(complete_path_map, positions, cut_idx)
		cut_qubit, _ = cut
		path = complete_path_map[cut_qubit]
		# print('start_frag_idx = ', start_frag_idx)
		# print('path = ', path)
		# start_frag = path[start_frag_idx]
		start_frag = None
		for x in path:
			if x[0] == start_frag_idx:
				start_frag = x
		print('start_frag_idx = %d, end_frag_idx = %d' % (start_frag_idx, end_frag_idx))
		for s_idx, s in enumerate(all_s):
			key = ''
			for char in s:
				key += str(char)
			fragment_s = fragment_all_s[key]
			# print('fragment_s = ', fragment_s)
			for sub_circ_measure, sub_circ_registers in fragment_s:
				sub_circ_idx = sub_circ_registers[0][0]
				print(Fore.RED + 'cut(i) = %d, circ_config_idx (j) = %d, s = %s, fragment (k) = %d' %(cut_idx,s_idx,s,sub_circ_idx) + Style.RESET_ALL)
				sub_circ_prob_sum = 0
				for x in sub_circ_measure:
					sub_circ_prob_sum += sub_circ_measure[x]
				print('old probs:', sub_circ_measure, 'old probs sum = ', sub_circ_prob_sum)
				if sub_circ_idx == end_frag_idx:
					print('sub_circ_idx = end_frag_index = ', sub_circ_idx)
					if s[cut_idx] == 4 or s[cut_idx] == 6 or s[cut_idx] == 8:
						for key in sub_circ_measure:
							print('multiply -0.5 for key ', key)
							sub_circ_measure[key] *= -0.5
					else:
						for key in sub_circ_measure:
							print('multiply +0.5 for key ', key)
							sub_circ_measure[key] *= 0.5
				elif sub_circ_idx == start_frag_idx:
					print('sub_circ_idx = start_frag_index = ', sub_circ_idx)
					if s[cut_idx] != 1 and s[cut_idx] != 2:
						# print('sub_circ_measure = ', sub_circ_measure)
						# print('sub_circ_registers = ', sub_circ_registers)
						# print('start_frag = ', start_frag)
						for key in sub_circ_measure:
							for register_idx, register_measurement in enumerate(key.split(' ')):
								_, sub_circ_register = sub_circ_registers[register_idx]
								if sub_circ_register == start_frag[2][0] and register_measurement[start_frag[2][1]] == '1':
									print('multiply -1 for key', key)
									sub_circ_measure[key] *= -1

				print('modified sub_circ_measure = ', sub_circ_measure)
			print('new fragment_s : ', fragment_s, '\n')
	print('fragment_all_s = ')
	print(fragment_all_s)
	return fragment_all_s

def fragment_combiner(frag_a, frag_b):
	# print('combine : ')
	# print(frag_0)
	# print(frag_1)
	combined_a_b = {}

	a_y_sigma_states = frag_a[0]
	a_registers = frag_a[1]
	b_y_sigma_states = frag_b[0]
	b_registers = frag_b[1]

	combined_registers = a_registers + b_registers

	for frag_a_state in a_y_sigma_states:
		for frag_b_state in b_y_sigma_states:
			if frag_a_state != '':
				key = frag_a_state + ' ' + frag_b_state
			else:
				key = frag_a_state + frag_b_state
			prob = a_y_sigma_states[frag_a_state] * b_y_sigma_states[frag_b_state]
			combined_a_b[key] = prob
	
	combined_fragment = (combined_a_b, combined_registers)
	# print('combined_fragment = ', combined_fragment)
	return combined_fragment

def combiner(fragment_all_s):
	print('*' * 200)
	print('combining fragment_all_s')
	empty_dict = {}
	empty_dict[''] = 1
	for s in fragment_all_s:
		combined_fragment_s = (empty_dict, [])
		fragment_s = fragment_all_s[s]
		for frag_to_combine in fragment_s:
			combined_fragment_s = fragment_combiner(combined_fragment_s, frag_to_combine)
		fragment_all_s[s] = combined_fragment_s
	# print('combined fragment_all_s: ', fragment_all_s)
	return fragment_all_s

def reconstructor(fragment_all_s, complete_path_map):
	reconstructed_prob = {}
	empty_y = []
	for i in range(len(complete_path_map)):
		empty_y.append('x')
	for s in fragment_all_s:
		y_sigma_distribution_s = fragment_all_s[s][0]
		registers_s = fragment_all_s[s][1]
		for y_sigma in y_sigma_distribution_s:
			prob = y_sigma_distribution_s[y_sigma]
			y_state = empty_y
			for register_idx, register_measure in enumerate(y_sigma.split(' ')):
				if 'measure' not in registers_s[register_idx][1].name:
					for char_idx in range(len(register_measure)):
						char = register_measure[char_idx]
						input_qubit = input_qubit_locator(complete_path_map, 
						registers_s[register_idx][0],
						registers_s[register_idx][1][char_idx])
						input_qubit_location = input_qubit[1]
						# print('y_state digit %d should be ' % input_qubit_location, char)
						y_state[input_qubit_location] = char
			y_state_str = ''
			for char in y_state:
				y_state_str += char
			if y_state_str in reconstructed_prob:
				reconstructed_prob[y_state_str] += prob
			else:
				reconstructed_prob[y_state_str] = prob
	return reconstructed_prob

def input_qubit_locator(complete_path_map, sub_circ_idx, qubit):
	# print('looking for qubit ', qubit, 'in sub_circ', sub_circ_idx)
	for input_qubit in complete_path_map:
		path = complete_path_map[input_qubit]
		output_sub_circ_idx = path[len(path)-1][0]
		output_qubit = path[len(path)-1][2]
		if sub_circ_idx == output_sub_circ_idx and qubit == output_qubit:
			return input_qubit
	return None

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
			# single_shot_tn_val[output_state] = np.power(8,len(positions)) * c_s * t_s[output_state]
			single_shot_tn_val[output_state] = c_s * t_s[output_state]
		print('single shot tensor network value = ', single_shot_tn_val)
		print('*' * 200)
		for key in single_shot_tn_val:
			if key in cumulative_tn_val:
				cumulative_tn_val[key] += single_shot_tn_val[key]
			else:
				cumulative_tn_val[key] = single_shot_tn_val[key]
		num_trials += 1
	# for output_state in cumulative_tn_val:
	# 	cumulative_tn_val[output_state] /= num_trials
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
	# circ = pickle.load(open('%s/supremacy_circuit_4_8.dump' % path, 'rb' ))
	# circ.draw(output='text',line_length = 400, filename='%s/original_circ.txt' % path)
	print('Original circuit:')
	print(circ)

	original_dag = circuit_to_dag(circ)
	dag_drawer(original_dag, filename='%s/original_dag.pdf' % path)
	q = circ.qregs[0]
	
	''' Test positions for the toy circuit'''
	# positions = [(q[4],1), (q[1],1)]
	# positions = [(q[1],0), (q[0],0)]
	# positions = [(q[1],0)]
	# positions = [(q[2],0), (q[1],0), (q[0],0)]
	# positions = [(q[0],0)]
	positions = [(q[0],0), (q[1],0)]

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
	c = ClassicalRegister(2, 'c')
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
		# new_key = key
		counts_rev[new_key] = counts[key] / 1024
	
	# result = tensor_network_val_calc(sub_circs_no_bridge, complete_path_map, positions, 10, post_process_fn=None)

	fragment_all_s, all_s = fragment_all_s_calc(sub_circs_no_bridge, complete_path_map, positions)
	fragment_all_s = coefficients_multiplier(fragment_all_s, positions, all_s, complete_path_map)
	fragment_all_s = combiner(fragment_all_s)
	reconstructed_prob = reconstructor(fragment_all_s, complete_path_map)

	print('Measure original circuit:')
	print(qc)
	print('original circ output prob = ', counts_rev)
	print('reconstructed prob = ', reconstructed_prob)
	print('-' * 200)


	# cut_circ_total_prob = 0
	# for x in result.values():
	# 	cut_circ_total_prob += x
	# print('with cutting:', result)
	# print('sum of probabilities = ', cut_circ_total_prob)

if __name__ == '__main__':
	main()