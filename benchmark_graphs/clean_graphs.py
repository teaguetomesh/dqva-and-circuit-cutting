import glob
import networkx as nx
import sys, os
import pickle

dqva_dir = '/'.join(os.path.abspath(os.getcwd()).split('/')[:-1])
sys.path.append(dqva_dir)

from utils.graph_funcs import graph_from_file

graph_files = glob.glob('*graphs')

for gf in graph_files:
    print(gf)
    num_total = 0
    num_removed = 0
    graphs = glob.glob(gf + '/G*')
    for graph in graphs:
        num_total += 1

        G = graph_from_file(dqva_dir + '/benchmark_graphs/' + graph)
        if not nx.is_connected(G):
            num_removed += 1
            os.remove(graph)

    print('num_total:', num_total)
    print('num_removed:', num_removed)
    print()

