"""
Use the benchmark graphs to test the performance of QAOA+
"""
import os, sys, argparse, glob
import numpy as np
from ansatz import qaoa_plus
import pickle, random
from utils.graph_funcs import *
from utils.helper_funcs import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', type=str, default=None,
                        help='path to dqva project')
    parser.add_argument('--alg', type=str, default='QAOA+',
                        help='name of algorithm to use')
    parser.add_argument('--graph', type=str, default=None,
                        help='glob path to the benchmark graph(s)')
    parser.add_argument('-P', type=int, default=1,
                        help='P-value for algorithm')
    parser.add_argument('--sim', type=str, default='qasm',
                        help='Choose the simulation backend')
    #parser.add_argument('--reps', type=int, default=4,
    #                    help='Number of repetitions to run')
    #parser.add_argument('-m', type=int, default=3,
    #                    help='Number of mixer rounds')
    #parser.add_argument('--shots', type=int, default=8192,
    #                    help='Number of shots')
    parser.add_argument('-v', type=int, default=1,
                        help='verbose')
    #parser.add_argument('--plim', type=int, default=None,
    #                    help='Limit the number of parameters')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    DQVAROOT = args.path
    if DQVAROOT[-1] != '/':
        DQVAROOT += '/'
    sys.path.append(DQVAROOT)

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
        print(graphname, G.edges())
        nq = len(G.nodes)

        data_list = []
        for Lambda in np.arange(0.1, 10, 0.7):
            data_dict = {'lambda':Lambda, 'graph':graphfn}
            out = qaoa_plus.solve_mis(args.P, G, Lambda)

            # Compute the approximation ratio by pruning the resulting measurements
            ratio = qaoa_plus.get_approximation_ratio(out, args.P, G)
            data_dict['ratio'] = ratio

            ranked_probs = qaoa_plus.get_ranked_probs(args.P, G, out['x'])
            for i, rp in enumerate(ranked_probs):
                if rp[2]:
                    data_dict['rank'] = i+1
                    data_dict['prob'] = rp[1]*100
                    break

            if 'rank' not in data_dict.keys():
                data_dict['rank'] = -1
                data_dict['prob'] = 0

            print('lambda: {:.3f}, ratio = {:.3f}, rank = {}, prob = {:.3f}'.format(
                      Lambda, ratio, data_dict['rank'], data_dict['prob']))

            data_list.append(data_dict)

        # Save the results
        savename = '{}_{}_P{}_{}.pickle'.format(graphname, args.alg, args.P,
                                                args.sim)

        with open(cur_savepath+savename, 'ab') as pf:
            pickle.dump(data_list, pf)

if __name__ == '__main__':
    main()

