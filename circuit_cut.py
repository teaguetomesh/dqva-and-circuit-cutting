from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
from qiskit.extensions.standard import CHGate, HGate, CnotGate, CyGate, CzGate
from qiskit.circuit import Measure
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit import BasicAer
from qiskit.visualization import plot_histogram
from help_fun import cut_edge, generate_sub_circs
import numpy as np
import networkx as nx
import pickle
import argparse
import os

def foo(args):
    path = '%s/results' % args.home_dir
    if not os.path.isdir(path):
        os.makedirs(path)

    q = QuantumRegister(4, 'q')
    # c = ClassicalRegister(4, 'c')
    circ = QuantumCircuit(q)
    circ.h([q[2],q[1]])
    circ.x([q[2],q[0]])
    circ.y(q[3])
    circ.cx(q[0], q[1])
    circ.cy(q[2], q[3])
    circ.ch(q[1], q[2])
    circ.x(q[2])
    # circ.barrier()
    # circ.measure(q,c)
    circ.draw()

    original_dag = circuit_to_dag(circ)
    print('original_dag has %d connected components' % nx.number_weakly_connected_components(original_dag._multi_graph))
    # print(nx.algorithms.connectivity.cuts.minimum_edge_cut(dag._multi_graph))
    dag_drawer(original_dag, filename='%s/original_dag.pdf' % path)
    circ.draw(output='mpl',filename='%s/original_circ.pdf' % path)

    cut_dag = cut_edge(original_dag=original_dag, wire=q[2],source_node_name='cy', dest_node_name='ch')
    print('cut_dag has %d connected components' % nx.number_weakly_connected_components(cut_dag._multi_graph))
    dag_drawer(cut_dag, filename='%s/cut_dag.pdf' % path)
    dag_to_circuit(cut_dag).draw(output='mpl',filename='%s/cut_circ.pdf' % path)

    sub_circs, _ = generate_sub_circs(cut_dag, q[2])
    for i, sub_circ in enumerate(sub_circs):
        dag_drawer(circuit_to_dag(sub_circ), filename='%s/sub_dag_%d.pdf' % (path, i))
        sub_circ.draw(output='mpl',filename='%s/sub_circ_%d.pdf' % (path, i))

def main():
    parser = argparse.ArgumentParser(description='Single circuit cut testing')
    parser.add_argument('--home-dir', type=str, default='.',
    help='home directory (default:.)')
    args = parser.parse_args()
    
    foo(args)

if __name__ == '__main__':
	main()