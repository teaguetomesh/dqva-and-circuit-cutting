#!/usr/bin/env python
import networkx as nx
import numpy as np
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=None,
                        help='Graph size')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    for i, graph_type in enumerate(['3_regular', '10_erdosrenyi',
                                    '20_2_2_community',
                                    '20_2_5_community']):
        print(graph_type)
        graph_size = args.n
        total_edges, cut_edges, relative_sizes = [], [], []
        count = 1
        while count <= 1000:
            # Generate a connected graph
            if i == 0:
                G = nx.generators.random_graphs.random_regular_graph(3, graph_size)
            elif i == 1:
                G = nx.erdos_renyi_graph(graph_size, 0.1)
                while not nx.is_connected(G):
                    G = nx.erdos_renyi_graph(graph_size, 0.1)
            elif i == 2:
                G = nx.generators.planted_partition_graph(l=2, k=int(graph_size/2), p_in=0.2, p_out=0.02)
                while not nx.is_connected(G):
                    G = nx.generators.planted_partition_graph(l=2, k=int(graph_size/2), p_in=0.2, p_out=0.02)
            elif i == 3:
                G = nx.generators.planted_partition_graph(l=5, k=int(graph_size/5), p_in=0.2, p_out=0.02)
                while not nx.is_connected(G):
                    G = nx.generators.planted_partition_graph(l=5, k=int(graph_size/5), p_in=0.2, p_out=0.02)

            total_edges.append(len(G.edges))
            # Partition graph
            setA, setB = nx.algorithms.community.kernighan_lin_bisection(G)

            if len(setA) >= len(setB):
                relative_sizes.append(len(setB) / len(setA))
            else:
                relative_sizes.append(len(setA) / len(setB))

            cutsetA = set(filter(lambda i: any([j in setB for j in G.neighbors(i)]), setA))
            cutsetB = set(filter(lambda i: any([j in setA for j in G.neighbors(i)]), setB))

            cutedges = [edge for edge in G.edges() if (edge[0] in setA and edge[1] in setB)
                                                       or (edge[0] in setB and edge[1] in setA)]
            cut_edges.append(len(cutedges))

            count += 1

        # Average results
        print(f'\tAveraged over 1000 random {graph_size}-node graphs: {np.mean(total_edges):.2f} total edges per graph,\n'
              f'\twith {np.mean(cut_edges):.2f} cut edges in the bisection,\n'
              f'\tand {np.mean(relative_sizes):.2f} ratio between the subgraphs\n')

        outdir = 'benchmark_results/graph_partition_statistics/'
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outfile = f'{outdir}{graph_size}node_{graph_type}_graphs.out'
        with open(outfile, 'w') as fn:
            fn.write(f'Average results over 1000 random {graph_size} node, {graph_type} graphs\n')
            fn.write(f'Total edges per graph: {np.mean(total_edges):.3f}   ({np.std(total_edges):.3f})\n')
            fn.write(f'Total cut edges: {np.mean(cut_edges):.3f}   ({np.std(cut_edges):.3f})\n')

if __name__ == '__main__':
    main()
