import argparse
import os
import pickle
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
import supremacy_generator as suprem_gen
import circuit_cut as cutter
import uniter as uniter

def main():
    path = './results'
    if not os.path.isdir(path):
            os.makedirs(path)
    
    # Generate a circuit
    circ = suprem_gen.circuit_generator(circuit_dimension=[4,4,8])

    # Cut a circuit
    q = circ.qregs[0]
    dag_drawer(circuit_to_dag(circ),filename='./results/original_dag.pdf')
    ''' Test positions that will cut into 2 parts'''
    positions = [(q[7],1),(q[2],1),(q[14],2),(q[10],1)]
    sub_circs_no_bridge, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
    print(sub_circs_no_bridge[0])
    print('K=%d, d=%d cluster' % (K,d))
    for i, sub_circ_no_bridge in enumerate(sub_circs_no_bridge):
        sub_circ_no_bridge.draw(output='text', line_length=400,filename='./results/sub_circ_%d.txt'%i)
    fragment_all_s = uniter.fragment_simulator(sub_circs_no_bridge, complete_path_map, positions)
    print(fragment_all_s)

if __name__ == '__main__':
	main()