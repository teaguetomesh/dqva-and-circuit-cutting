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

def fragment_s_calc(s, sub_circs_no_bridge, complete_path_map, num_shots=1024):
	sub_circs_bridge = sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map)
	backend_sim = BasicAer.get_backend('qasm_simulator')
	fragment_s = []

	for sub_circ_idx, sub_circ in enumerate(sub_circs_bridge):
		job_sim = execute(sub_circ, backend_sim, shots=num_shots)
		result_sim = job_sim.result()
		# print(type(result_sim), result_sim)
		counts = result_sim.get_counts(sub_circ)
		y_sigma_freq  = {}
		for key in counts:
			y_sigma_freq[key[::-1]] = counts[key]/num_shots
		[(sub_circ_idx, creg) for creg in sub_circ.cregs]
		fragment_s.append((y_sigma_freq, [(sub_circ_idx, creg) for creg in sub_circ.cregs]))
	return fragment_s

def fragment_all_s_calc(sub_circs_no_bridge, complete_path_map, positions):
    all_s = sequential_s(len(positions))
    fragment_all_s = {}
    # for i in range(np.power(8, len(positions))):
    print('Simulating fragments for %d s samples * %d fragment circuits' % 
    (len(all_s), len(sub_circs_no_bridge)))
    for s in all_s:
        key = ''
        for char in s:
            key += str(char)
        fragment_s = fragment_s_calc(s, sub_circs_no_bridge, complete_path_map)
        fragment_all_s[key] = fragment_s
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
		# print('start_frag_idx = %d, end_frag_idx = %d' % (start_frag_idx, end_frag_idx))
		for s_idx, s in enumerate(all_s):
			key = ''
			for char in s:
				key += str(char)
			fragment_s = fragment_all_s[key]
			# print('fragment_s = ', fragment_s)
			for sub_circ_measure, sub_circ_registers in fragment_s:
				sub_circ_idx = sub_circ_registers[0][0]
				# print(Fore.RED + 'cut(i) = %d, circ_config_idx (j) = %d, s = %s, fragment (k) = %d' %(cut_idx,s_idx,s,sub_circ_idx) + Style.RESET_ALL)
				sub_circ_prob_sum = 0
				for x in sub_circ_measure:
					sub_circ_prob_sum += sub_circ_measure[x]
				# print('old probs:', sub_circ_measure, 'old probs sum = ', sub_circ_prob_sum)
				if sub_circ_idx == end_frag_idx:
					# print('sub_circ_idx = end_frag_index = ', sub_circ_idx)
					if s[cut_idx] == 4 or s[cut_idx] == 6 or s[cut_idx] == 8:
						for key in sub_circ_measure:
							# print('multiply -0.5 for key ', key)
							sub_circ_measure[key] *= -0.5
					else:
						for key in sub_circ_measure:
							# print('multiply +0.5 for key ', key)
							sub_circ_measure[key] *= 0.5
				elif sub_circ_idx == start_frag_idx:
					# print('sub_circ_idx = start_frag_index = ', sub_circ_idx)
					if s[cut_idx] != 1 and s[cut_idx] != 2:
						# print('sub_circ_measure = ', sub_circ_measure)
						# print('sub_circ_registers = ', sub_circ_registers)
						# print('start_frag = ', start_frag)
						for key in sub_circ_measure:
							for register_idx, register_measurement in enumerate(key.split(' ')):
								_, sub_circ_register = sub_circ_registers[register_idx]
								if sub_circ_register == start_frag[2][0] and register_measurement[start_frag[2][1]] == '1':
									# print('multiply -1 for key', key)
									sub_circ_measure[key] *= -1

			# 	print('modified sub_circ_measure = ', sub_circ_measure)
			# print('new fragment_s : ', fragment_s, '\n')
	# print('fragment_all_s = ')
	# print(fragment_all_s)
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
	print('Combining fragments in each s sample')
	empty_dict = {}
	empty_dict[''] = 1
	for s in fragment_all_s:
		combined_fragment_s = (empty_dict, [])
		fragment_s = fragment_all_s[s]
		for frag_to_combine in fragment_s:
			combined_fragment_s = fragment_combiner(combined_fragment_s, frag_to_combine)
		fragment_all_s[s] = combined_fragment_s
	return fragment_all_s

def reconstructor(fragment_all_s, complete_path_map):
    print('Combining probabilities for all s samples')
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

def chiSquared(dist1,dist2):
    chi=0
    for key in dist1:
        chi=chi+((dist1[key]-dist2[key])**2/dist1[key])
    return chi