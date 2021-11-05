#!/usr/bin/env python
import sys, argparse, glob
import pickle
import networkx as nx
from pathlib import Path

import mis
import partition_no_cuts
from utils.graph_funcs import graph_from_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='path to dqva project')
    parser.add_argument('--graph', type=str, default=None,
                        help='glob path to the benchmark graph(s)')
    parser.add_argument('--numcuts', type=int, default=1,
                        help='Number of cuts')
    parser.add_argument('--rounds', type=int, default=4,
                         help='Number of partition rounds')
    parser.add_argument('--shots', type=int, default=8192,
                        help='Number of shots')
    parser.add_argument('--rep', type=int, default=1,
                        help='Rep number for labelling')
    parser.add_argument('--numfrags', type=int, default=2,
                        help='Number of subgraphs to generate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    DQVAROOT = args.path
    if DQVAROOT[-1] != '/':
        DQVAROOT += '/'
    sys.path.append(DQVAROOT)

    all_graphs = glob.glob(DQVAROOT + args.graph)

    graphsave = args.graph.split('/')[1].strip('_graphs')

    savepath = DQVAROOT + f'benchmark_results/ISCA_results/COBYLA/{graphsave}_{args.numfrags}frags_{args.numcuts}cuts_{args.shots}shots/'
    Path(savepath).mkdir(parents=True, exist_ok=True)

    for graphfn in all_graphs:
        graphname = graphfn.split('/')[-1].strip('.txt')
        cur_savepath = savepath + '{}/'.format(graphname)
        Path(cur_savepath).mkdir(parents=True, exist_ok=True)

        G = graph_from_file(graphfn)
        nq = len(G.nodes)

        print('Loaded graph: {}, with {} nodes'.format(graphfn, nq))
        print('Nodes:', G.nodes)
        print('Edges:', G.edges)

        init_state = '0'*G.number_of_nodes()
        full_history = []
        for rounds in range(args.rounds):
            print('-------------- ROUND {} BEGIN --------------\n\n'.format(rounds+1))
            if args.numcuts > 0:
                out = mis.solve_mis_cut_dqva(init_state, G, m=1, verbose=1,
                                        shots=args.shots, max_cuts=args.numcuts,
                                        num_frags=args.numfrags)
            else:
                out = partition_no_cuts.solve_mis_no_cut_dqva(init_state, G, m=1,
                                                    shots=args.shots, verbose=1)
            init_state = out[0]
            full_history.append(out)

        savefn = 'dqva_{}_{}cuts_rep{}.pickle'.format(graphname, args.numcuts, args.rep)
        with open(cur_savepath+savefn, 'wb') as pf:
            pickle.dump((G, full_history), pf)

if __name__ == '__main__':
    main()
