import numpy as np
from qiskit import *
from qiskit.visualization import *
import matplotlib.pyplot as plt
import argparse
from help_fun import *
import random
import pickle

def circuit_generator(args):
	row = args.circuit_dimension
	col = row
	q_reg = q_register(row = row, col = col)
	circ = QuantumCircuit(q_reg.qubits)

	circ = full_entangle(circuit = circ, q_reg = q_reg)
	# random_start = random.randint(1,8)
	random_start = 0
	# circ = cz_layer(circuit = circ, q_reg = q_reg, rotation_idx = random_start, single_qubit_gates = False)
	# for i in range(1,max(8, args.circuit_depth)):
	# 	circ = cz_layer(circuit = circ, q_reg = q_reg, rotation_idx = i+random_start, single_qubit_gates = True)

	for i in range(int(args.circuit_depth/8)):
		if i==0:
			circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 0, single_qubit_gates = False)
		else:
			circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 0, single_qubit_gates = False)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 2, single_qubit_gates = True)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 1, single_qubit_gates = True)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 3, single_qubit_gates = True)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 4, single_qubit_gates = True)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 6, single_qubit_gates = True)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 5, single_qubit_gates = True)
		circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = 7, single_qubit_gates = True)

	c_reg = ClassicalRegister(row*col, 'c')
	meas = QuantumCircuit(q_reg.qubits, c_reg)
	meas.barrier(q_reg.qubits)
	meas.measure(q_reg.qubits,c_reg)

	circ = circ+meas

	return circ

def main():
	parser = argparse.ArgumentParser(description='SKLearn classifiers for quantum readout')

	parser.add_argument('--circuit-dimension', type=int, default=4, metavar='N',
	help='dimension of a square quantum supremacy circuit')
	parser.add_argument('--circuit-depth', type=int, default=8, metavar='N',
	help='depth of a square quantum supremacy circuit')
	args = parser.parse_args()

	circ = circuit_generator(args)
	circ.draw(output='text', line_length = 400, filename = 'circ_draw_%d.txt' % args.circuit_dimension)
	pickle.dump(circ, open( "circuit.p", "wb" ))
	# style = {'figwidth': 20}
	# circ.draw(output='mpl', filename = 'circ_draw_%d.pdf' % args.circuit_dimension)

if __name__ == '__main__':
	main()