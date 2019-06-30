from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
from cutting_help_fun import *
import pickle
import networkx as nx
import numpy as np
import networkx as nx
import pickle
import argparse
import os

def foo(args):
    path = '%s/results' % args.home_dir
    if not os.path.isdir(path):
        os.makedirs(path)

    circ = pickle.load(open('results/supremacy_circuit_4_8.dump', 'rb' ))
    original_dag = circuit_to_dag(circ)
    q = circ.qregs[0]
    
    ''' Test positions that will cut into 3 parts'''
    # positions = [(q[2], 1), (q[7], 1), (q[8],3), (q[9],6), (q[10], 4), (q[10],1), (q[14], 1)]
    ''' Test positions that will cut into 2 parts'''
    positions = [(q[2], 1), (q[7], 1), (q[10],1), (q[14], 1)]

    cut_dag, path_order_dict = cut_edges(original_dag=original_dag, positions=positions)
    dag_drawer(cut_dag, filename='%s/cut_dag.pdf' % path)
    print('cut_dag has %d connected components' % nx.number_weakly_connected_components(cut_dag._multi_graph))

    in_out_arg_dict = contains_wire_nodes(cut_dag)
    sub_reg_dicts, input_wires_mapping = sub_circ_reg_counter(cut_dag, in_out_arg_dict)

    print('\ninput_wires_mapping:')
    [print(x, input_wires_mapping[x]) for x in input_wires_mapping]

    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    translation_dict = translation_dict_calc(input_wires_mapping, components, in_out_arg_dict, sub_reg_dicts)
    complete_path_map = complete_path_calc(path_order_dict, input_wires_mapping, translation_dict, sub_reg_dicts)
    print('\ncomplete_path_map:')
    [print(x, complete_path_map[x]) for x in complete_path_map]

    sub_circs = generate_sub_circs(cut_dag, positions)
    for idx, sub_circ in enumerate(sub_circs):
        sub_circ.draw(output='text',line_length = 400, filename='%s/sub_circ_%d.txt' % (path, idx))
        dag_drawer(circuit_to_dag(sub_circ), filename='%s/sub_dag_%d.pdf' % (path, idx))

def main():
    parser = argparse.ArgumentParser(description='Single circuit cut testing')
    parser.add_argument('--home-dir', type=str, default='.',
    help='home directory (default:.)')
    args = parser.parse_args()
    
    foo(args)

if __name__ == '__main__':
	main()