#!/usr/bin/env python
import glob
from pathlib import Path
import networkx as nx
import numpy as np

import qcopt


def random_independent_set(G):
    n = len(G.nodes)
    stopping_condition = [0] * n

    independent_set = []

    while sum(stopping_condition) < n:
        unvisited_nodes = [i for i, bit in enumerate(stopping_condition) if bit == 0]

        next_node = np.random.choice(unvisited_nodes)
        stopping_condition[next_node] = 1

        if not any([neighbor in independent_set for neighbor in G.neighbors(next_node)]):
            independent_set.append(next_node)

    mis_bitstr = ''.join(['1' if n in independent_set else '0' for n in sorted(list(G.nodes))])
    if not qcopt.graph_funcs.is_indset(mis_bitstr, G, little_endian=False):
        raise Exception('Produced an invalid independent set!')

    return len(independent_set)


all_graph_types = glob.glob('benchmark_graphs/N*graphs')

for graph_type in all_graph_types:
    all_graphs = glob.glob(f'{graph_type}/G*.txt')
    savepath = f'benchmark_results/randomized_mis/{graph_type.split("/")[-1]}'
    Path(savepath).mkdir(parents=True, exist_ok=True)
    for graph in all_graphs:
        G = qcopt.graph_funcs.graph_from_file(graph)
        rand_mis = []
        for _ in range(5):
            rand_mis.append(random_independent_set(G))
        print(f'{"/".join(graph.split("/")[-2:])} avg random mis size: {np.mean(rand_mis)}')
        with open(f'{savepath}/{graph.split("/")[-1].strip(".txt")}_rand_results.txt', 'w') as fn:
            fn.write(f'Average random mis size over 5 repetitions: {np.mean(rand_mis)}')
