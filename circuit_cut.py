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

    circ = pickle.load(open('%s/supremacy_circuit_4_8.dump' % path, 'rb' ))
    original_dag = circuit_to_dag(circ)
    q = circ.qregs[0]
    positions = [(q[2], 1), (q[7], 1), (q[10], 1), (q[14], 1)]
    wires_being_cut = [x[0] for x in positions]

    cut_dag, path_order_dict = cut_edges(original_dag=original_dag, positions=positions)
    in_out_arg_dict = contains_wire_nodes(cut_dag)
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    sub_reg_dicts, input_wires_mapping = reg_dict_counter(cut_dag, wires_being_cut)
    translation_dict = translation_dict_calc(input_wires_mapping, components, in_out_arg_dict, sub_reg_dicts)
    complete_path_map = complete_path_calc(path_order_dict, input_wires_mapping, translation_dict, sub_reg_dicts)

    [print(x, complete_path_map[x]) for x in complete_path_map]
    sub_circs = generate_sub_circs(cut_dag, positions)
    for sub_circ_idx, sub_circ in enumerate(sub_circs):
        dag_drawer(circuit_to_dag(sub_circ), filename='%s/sub_dag_%d.pdf' % (path, sub_circ_idx))
        sub_circ.draw(output='text',line_length = 400, filename='%s/sub_circ_%d.txt' % (path, sub_circ_idx))

def main():
    parser = argparse.ArgumentParser(description='Single circuit cut testing')
    parser.add_argument('--home-dir', type=str, default='.',
    help='home directory (default:.)')
    args = parser.parse_args()
    
    foo(args)

if __name__ == '__main__':
	main()