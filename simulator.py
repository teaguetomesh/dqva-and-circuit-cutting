import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit import BasicAer, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.tools.visualization import dag_drawer
# from qiskit.circuit import Measure
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
import os
import pickle
import random
import copy
import timeit

def simulate_fragments(sub_circs_no_bridge, complete_path_map):
	fragment_all_s, all_s, sub_circ_qubits = fragment_all_s_calc(sub_circs_no_bridge, complete_path_map, random_sampling=False)
	fragment_all_s = coefficients_multiplier(fragment_all_s, complete_path_map)
	return fragment_all_s

def sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map):
	s_idx = 0
	sub_circs_bridge = copy.deepcopy(sub_circs_no_bridge)

	for input_qubit in complete_path_map:
		path = complete_path_map[input_qubit]
		num_bridges = len(path) - 1
		for bridge_idx in range(num_bridges):
			sample_s = s[s_idx]
			s_idx += 1
			source_circ_idx = path[bridge_idx][0]
			dest_circ_idx = path[bridge_idx+1][0]
			source_circ_dag = circuit_to_dag(sub_circs_bridge[source_circ_idx])
			dest_circ_dag = circuit_to_dag(sub_circs_bridge[dest_circ_idx])
			source_qubit = path[bridge_idx][1]
			dest_qubit = path[bridge_idx+1][1]
			if sample_s == 2:
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_qubit],
				cargs=[])
			if sample_s == 3:
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[source_qubit])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_qubit],
				cargs=[])
			if sample_s == 4:
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[source_qubit])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_qubit],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_qubit],
				cargs=[])
			if sample_s == 5:
				source_circ_dag.apply_operation_back(op=SdgGate(), 
				qargs=[source_qubit])
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[source_qubit])
				dest_circ_dag.apply_operation_front(op=SGate(),
				qargs=[dest_qubit],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_qubit],
				cargs=[])
			if sample_s == 6:
				source_circ_dag.apply_operation_back(op=SdgGate(), 
				qargs=[source_qubit])
				source_circ_dag.apply_operation_back(op=HGate(), 
				qargs=[source_qubit])
				dest_circ_dag.apply_operation_front(op=SGate(),
				qargs=[dest_qubit],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=HGate(),
				qargs=[dest_qubit],
				cargs=[])
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_qubit],
				cargs=[])
			if sample_s == 8:
				dest_circ_dag.apply_operation_front(op=XGate(),
				qargs=[dest_qubit],
				cargs=[])
			sub_circs_bridge[source_circ_idx] = dag_to_circuit(source_circ_dag)
			sub_circs_bridge[dest_circ_idx] = dag_to_circuit(dest_circ_dag)

	return sub_circs_bridge

def fragment_s_calc(s, sub_circs_no_bridge, complete_path_map):
	# print('simulating for s =',s)
	sub_circs_bridge = sub_circ_sampler(s, sub_circs_no_bridge, complete_path_map)
	backend = BasicAer.get_backend('statevector_simulator')
	fragment_prob = []

	for sub_circ_idx, sub_circ in enumerate(sub_circs_bridge):
		# print('sub circ bridge %d\n'%sub_circ_idx)
		# print(sub_circ)
		job = execute(sub_circ, backend)
		result = job.result()
		outputstate = result.get_statevector(sub_circ)
		outputprob = [np.power(abs(x),2) for x in outputstate]

		# print('outputprob = ', len(outputprob))
		# print('*'*100)
		fragment_prob.append(outputprob)
	return fragment_prob

def fragment_all_s_calc(sub_circs_no_bridge, complete_path_map, random_sampling=False):
	num_cuts = 0
	for x in complete_path_map:
		num_cuts += len(complete_path_map[x])-1
	all_s = None
	if not random_sampling:
		all_s = sequential_s(num_cuts)
	else:
		all_s = random_s(num_cuts, np.power(8,num_cuts))
	
	sub_circ_qubits = []
	for sub_circ in sub_circs_no_bridge:
		sub_circ_qubits.append(sub_circ.qubits[::-1])
	
	fragment_all_s = {}
	print('Simulating fragments for %d s samples * %d fragment circuits' % 
	(len(all_s), len(sub_circs_no_bridge)))
	start = timeit.default_timer()
	for s_idx, s in enumerate(all_s):
		key = ''
		for char in s:
			key += str(char)
		fragment_prob = fragment_s_calc(s, sub_circs_no_bridge, complete_path_map)
		# print('s = ', s, 'fragment_prob =', np.shape(fragment_prob))
		fragment_all_s[key] = fragment_prob
		if s_idx%50 == 49:
			stop = timeit.default_timer()
			time_remaining = (stop-start)/(s_idx/len(all_s))-(stop-start)
			print('%.2f %% completed, estimated time remaining = %.2f seconds' % ((100*s_idx/len(all_s)),time_remaining))
	stop = timeit.default_timer()
	print('Total Time: %.2f seconds' % (stop - start))
	print('*'*100)
	return fragment_all_s, all_s, sub_circ_qubits

def coefficients_multiplier_helper(qubits):
	# print('qubits are:', qubits)
	cutQ_idx = []
	for idx, qubit in enumerate(qubits):
		if qubit[0].name == 'cutQ':
			cutQ_idx.append(idx)
	# print('cutQ qubits are at:', cutQ_idx)
	positions = []
	for state in range(np.power(2,len(qubits))):
		measurement = bin(state)[2:].zfill(len(qubits))
		# print('state = {}, measurement = {}'.format(state, measurement))
		sigma = 1
		for idx in cutQ_idx:
			if measurement[idx] == '1':
				sigma *= -1
		if sigma == -1:
			positions.append(state)
			# print('sigma=-1')
	return positions

def coefficients_multiplier(fragment_all_s, complete_path_map):
	print('Multiplying sigma and c for %d s combinations' % len(fragment_all_s))
	cuts = []
	for input_qubit in complete_path_map:
		path = complete_path_map[input_qubit]
		num_cuts = len(path)-1
		for i in range(num_cuts):
			start_frag_idx, start_qubit = path[i]
			end_frag_idx, end_qubit = path[i+1]
			cuts.append((start_frag_idx, start_qubit, end_frag_idx, end_qubit))
	for s_key in fragment_all_s:
		s = []
		for char in s_key:
			s.append(int(char))
		for cut_idx, cut in enumerate(cuts):
			start_frag_idx, _, end_frag_idx, _ = cut
			cut_s = s[cut_idx]
			l = fragment_all_s[s_key][start_frag_idx][0]
			if cut_s == 4 or cut_s == 6 or cut_s == 8:
				fragment_all_s[s_key][start_frag_idx] = ([x *(-0.5) for x in l], fragment_all_s[s_key][start_frag_idx][1])
			else:
				fragment_all_s[s_key][start_frag_idx] = ([x *(0.5) for x in l], fragment_all_s[s_key][start_frag_idx][1])
			
			l = fragment_all_s[s_key][end_frag_idx][0]
			qubits = fragment_all_s[s_key][end_frag_idx][1]
			positions = coefficients_multiplier_helper(qubits)
			if cut_s != 1 and cut_s != 2:
				fragment_all_s[s_key][end_frag_idx] = ([l[i]*(-1) if i in positions else l[i] for i in range(len(l))], 
				fragment_all_s[s_key][end_frag_idx][1])
				
	return fragment_all_s

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

def random_s(length, num_samples):
	all_s = sequential_s(length)
	print('all_s = ', len(all_s), num_samples)
	s_indices = np.random.choice(len(all_s),min(num_samples,len(all_s)),replace=False)
	s = []
	for i in s_indices:
		s.append(all_s[i])
	return s