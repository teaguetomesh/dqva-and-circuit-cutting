import argparse
import os
import pickle
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
import supremacy_generator as suprem_gen
import circuit_cut as cutter

def main():
    path = './results'
    if not os.path.isdir(path):
            os.makedirs(path)
    # Generate a circuit
    circ = suprem_gen.circuit_generator(circuit_dimension=[4,4,8])
    # print(circ)

    # Cut a circuit
    q = circ.qregs[0]
    dag_drawer(circuit_to_dag(circ),filename='./results/original_dag.pdf')
    ''' Test positions that will cut into 2 parts'''
    # positions = [(q[7],1),(q[2],1),(q[14],2),(q[10],1)]
    positions = [(q[11], 2), (q[10], 5), (q[9], 4), (q[8], 4)]
    cut_dag, path_order_dict = cutter.cut_edges(original_dag=circuit_to_dag(circ), positions=positions)
    dag_drawer(cut_dag,filename='./results/cut_dag.pdf')
    # sub_circs_no_bridge, complete_path_map = cutter.cut_circuit(circ, positions)
    # for i, sub_circ_no_bridge in enumerate(sub_circs_no_bridge):
    #     sub_circ_no_bridge.draw(output='text', line_length=400,filename='./results/sub_circ_%d.txt'%i)

if __name__ == '__main__':
	main()