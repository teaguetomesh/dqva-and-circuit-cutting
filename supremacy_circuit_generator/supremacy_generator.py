import numpy as np
import argparse
from supremacy_help_fun import full_entangle, supremacy_layer, q_register
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
import random
import pickle
import os

def circuit_generator(args):
    row = args.circuit_dimension[0]
    col = args.circuit_dimension[1]
    q_reg = q_register(row = row, col = col)
    circ = QuantumCircuit(q_reg.qubits)

    circ = full_entangle(circuit = circ, q_reg = q_reg)
    if not args.random:
        for idx in range(args.circuit_depth):
            order = args.layer_order[idx%8]
            if idx==0:
                circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = order, single_qubit_gates = False)
            else:
                circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = order, single_qubit_gates = True)

    else:
        random_start = random.randint(1,8)
        for idx in range(args.circuit_depth):
            if idx==0:
                circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = random_start, single_qubit_gates = False)
            else:
                circ = supremacy_layer(circuit = circ, q_reg = q_reg, rotation_idx = (random_start+idx)%8, single_qubit_gates = True)

    if args.measure:
        c_reg = ClassicalRegister(row*col, 'c')
        meas = QuantumCircuit(q_reg.qubits, c_reg)
        meas.barrier(q_reg.qubits)
        meas.measure(q_reg.qubits,c_reg)
        circ = circ+meas

    return circ

def main():
    parser = argparse.ArgumentParser(description='Google quantum supremacy circuit generator')

    parser.add_argument('--circuit-dimension', type=int, nargs="+", default=[4,4], metavar='N',
    help='dimension of a square quantum supremacy circuit (default: 4 by 4)')
    parser.add_argument('--circuit-depth', type=int, default=8, metavar='N',
    help='depth of a square quantum supremacy circuit (default:8)')
    parser.add_argument('--layer-order', type=int, default=[0,2,1,3,4,6,5,7],
    help='ordering of the layers in the supremacy circuit')
    parser.add_argument('--random', action='store_true', default=False,
    help='randomize layer orderings?')
    parser.add_argument('--measure', action='store_true', default=False,
    help='add measurements?')
    parser.add_argument('--home-dir', type=str, default='..',
    help='home directory (default:..)')
    args = parser.parse_args()

    circ = circuit_generator(args)

    path = '%s/results' % args.home_dir
    if not os.path.isdir(path):
            os.makedirs(path)
    circ.draw(output='text', line_length = 400, filename = '%s/supremacy_circ_%d*%d_%d.txt' % 
    (path, args.circuit_dimension[0], args.circuit_dimension[1], args.circuit_depth))
    dag_drawer(circuit_to_dag(circ), filename='%s/supremacy_dag_%d*%d_%d.pdf' %
    (path, args.circuit_dimension[0], args.circuit_dimension[1], args.circuit_depth))
    pickle.dump(circ, open('%s/supremacy_circuit_%d*%d_%d.dump' %(path, args.circuit_dimension[0],args.circuit_dimension[1], args.circuit_depth), 'wb' ))

if __name__ == '__main__':
	main()