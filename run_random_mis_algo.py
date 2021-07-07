#!/usr/bin/env python

import argparse
import glob
import os
import sys
import networkx as nx
import numpy as np

from utils.graph_funcs import graph_from_file, is_indset
from utils.helper_funcs import hamming_weight

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default=None,
                        help='glob path to the benchmark graph(s)')
    parser.add_argument('--reps', type=int, default=1,
                        help='num reps')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    all_graphs = glob.glob(args.graph)

    savepath = 'benchmark_results/random_mis/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    graphtype = args.graph.split('/')[1]
    savepath += graphtype + '/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    for graphfn in all_graphs:
        graphname = graphfn.split('/')[-1].strip('.txt')
        cur_savepath = savepath + '{}/'.format(graphname)
        if not os.path.isdir(cur_savepath):
            os.mkdir(cur_savepath)

        G = graph_from_file(graphfn)
        nq = len(G.nodes)

        print('Loaded graph: {}, with {} nodes'.format(graphfn, nq))

        for rep in range(1, args.reps+1):
            cur_mis = ['0'] * len(G.nodes)
            visited = [0] * len(G.nodes)

            counter = 0
            while sum(visited) < len(G.nodes):
                unvisited = [i for i, val in enumerate(visited) if val == 0]

                new_node = np.random.choice(unvisited)

                neighbors = list(G.neighbors(new_node))
                neighbor_vals = [int(cur_mis[n]) for n in neighbors]
                if sum(neighbor_vals) == 0:
                    cur_mis[new_node] = '1'

                visited[new_node] = 1

            mis = ''.join(cur_mis)
            valid = is_indset(''.join(cur_mis[::-1]), G)
            if valid:
                print('\tRep {}, found MIS with size {}'.format(rep, hamming_weight(mis)))
            else:
                raise Exception('MIS WAS NOT VALID!!')

            savefn = 'random_mis_rep{}.txt'.format(rep)
            with open(cur_savepath+savefn, 'w') as fn:
                fn.write(graphfn + '\n')
                fn.write(mis + '\n')
                fn.write(str(hamming_weight(mis)) + '\n')

if __name__ == '__main__':
    main()

