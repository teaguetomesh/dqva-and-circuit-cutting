"""
Use the benchmark graphs to test ability of the different ansatzes
to solve the MIS problem
"""
import os, sys, argparse, glob
import numpy as np
import dqva
import pickle
from utils.graph_funcs import graph_from_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', type=str, default=None,
                        help='path to dqva project')
    parser.add_argument('--alg', type=str, default=None,
                        help='name of algorithm to use')
    parser.add_argument('--graph', type=str, default=None,
                        help='glob path to the benchmark graph(s)')
    parser.add_argument('-P', type=int, default=1,
                        help='P-value for algorithm')
    parser.add_argument('--sim', type=str, default=None,
                        help='Choose the simulation backend')
    parser.add_argument('--reps', type=int, default=4,
                        help='Number of repetitions to run')
    parser.add_argument('-m', type=int, default=3,
                        help='Number of mixer rounds')
    parser.add_argument('--shots', type=int, default=8192,
                        help='Number of shots')
    parser.add_argument('-v', type=int, default=1,
                        help='verbose')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    DQVAROOT = args.path
    if DQVAROOT[-1] != '/':
        DQVAROOT += '/'
    sys.path.append(DQVAROOT)

    if args.alg not in ['qaoa', 'qva', 'dqva', 'cut_dqva']:
        raise Exception('Unknown algorithm:', args.alg)
    if args.sim not in ['qasm', 'statevector', 'cloud']:
        raise Exception('Unknown backend:', args.sim)

    all_graphs = glob.glob(DQVAROOT + args.graph)
    graph_type = all_graphs[0].split('/')[-2]

    savepath = DQVAROOT+'benchmark_results/{}_P{}_{}/'.format(args.alg, args.P, args.sim)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    savepath += '{}/'.format(graph_type)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    for graphfn in all_graphs:
        graphname = graphfn.split('/')[-1].strip('.txt')
        cur_savepath = savepath + '{}/'.format(graphname)
        if not os.path.isdir(cur_savepath):
            os.mkdir(cur_savepath)

        G = graph_from_file(graphfn)
        nq = len(G.nodes)
        init_state = '0'*nq

        for rep in range(1, args.reps+1):
            if args.alg == 'qaoa':
                out = dqva.solve_mis_qaoa(init_state, G, P=args.P, m=args.m,
                                          sim=args.sim, shots=args.shots,
                                          verbose=args.v)
            elif args.alg == 'qva':
                out = dqva.solve_mis_qva(init_state, G, P=args.P, m=args.m,
                                         sim=args.sim, shots=args.shots,
                                         verbose=args.v)
            elif args.alg == 'dqva':
                out = dqva.solve_mis_dqva(init_state, G, P=args.P, m=args.m,
                                          sim=args.sim, shots=args.shots,
                                          verbose=args.v)
            elif args.alg == 'cut_dqva':
                out = dqva.solve_mis_cut_dqva()

            savename = '{}_{}_P{}_{}_rep{}.pickle'.format(graphname, args.alg,
                                                          args.P, args.sim, rep)
            with open(cur_savepath+savename, 'ab') as pf:
                pickle.dump({'graph':graphfn, 'out':out}, pf)

if __name__ == '__main__':
    main()

