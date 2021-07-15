"""
Use the benchmark graphs to test ability of the different ansatzes
to solve the MIS problem
"""
import os, sys, argparse, glob
import numpy as np
import mis
import pickle, random
from utils.graph_funcs import graph_from_file, is_indset

def get_hw_1_strs(nq):
    bitstrs = []
    for i in range(nq-1):
        bitstr = list('0'*nq)
        bitstr[i] = '1'
        bitstr = ''.join(bitstr)
        bitstrs.append(bitstr)
    return bitstrs

def get_hw_2_strs(nq):
    bitstrs = []
    for i in range(nq-1):
        for j in range(i+1, nq):
            bitstr = list('0'*nq)
            bitstr[i] = '1'
            bitstr[j] = '1'
            bitstr = ''.join(bitstr)
            bitstrs.append(bitstr)
    return bitstrs

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
    parser.add_argument('--plim', type=int, default=None,
                        help='Limit the number of parameters')
    parser.add_argument('--extend', type=int, default=0,
                        help='Flag for whether to overwrite or extend repetitions')
    parser.add_argument('--startrep', type=int, default=0,
                        help='Integer repetition label')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of parallel threads passed to Aer')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    DQVAROOT = args.path
    if DQVAROOT[-1] != '/':
        DQVAROOT += '/'
    sys.path.append(DQVAROOT)

    if args.alg not in ['qaoa', 'dqva', 'qls', 'cut_dqva',
                        'qaoaHotStart', 'dqvaHotStart', 'qlsHotStart',
                        'qaoaWStart']:
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
        print(G.edges())
        nq = len(G.nodes)
        if 'HotStart' in args.alg:
            # Randomly select 3 bitstrings
            if nq < 8:
                all_bitstrs = get_hw_1_strs(nq)
            else:
                all_bitstrs = get_hw_2_strs(nq)

            init_states = []
            while len(init_states) < 3:
                temp_str = random.choice(all_bitstrs)
                # Ensure the randomly select bitstr is a viable IS
                if not is_indset(temp_str, G):
                    continue
                # Make sure it is not already added to the list
                overlap = False
                for bitstr in init_states:
                    #for x, y in zip(temp_str, bitstr):
                    #    if x == '1' and y == '1':
                    #        overlap = True
                    if bitstr == temp_str:
                        overlap = True
                if not overlap:
                    init_states.append(temp_str)
            print('init states:', init_states)
        elif 'WStart' in args.alg:
            init_state = 'W'
        else:
            init_state = '0'*nq

        if args.extend:
            print('savepath:', cur_savepath)
            all_reps = glob.glob(cur_savepath + '*rep*')
            print('{} reps completed'.format(len(all_reps)))
            if args.startrep > 0:
                rep_range = range(args.startrep, args.startrep+args.reps)
            elif len(all_reps) < args.reps:
                rep_range = range(len(all_reps)+1, args.reps+1)
            else:
                print('Skipping graph {}'.format(graphname))
                continue
            print('rep_range =', list(rep_range))
        else:
            rep_range = range(1, args.reps+1)

        for rep in rep_range:
            if args.alg == 'qaoa' or args.alg == 'qaoaWStart':
                out = mis.solve_mis_qaoa(init_state, G, P=args.P, m=args.m,
                                          sim='aer', shots=args.shots,
                                          verbose=args.v, threads=args.threads)
            elif args.alg == 'dqva':
                out = mis.solve_mis_dqva(init_state, G, P=args.P, m=args.m,
                                         sim='aer', shots=args.shots,
                                         verbose=args.v, threads=args.threads)
            elif args.alg == 'qls':
                out = mis.solve_mis_qls(init_state, G, P=args.P, m=args.m,
                                          sim='aer', shots=args.shots,
                                          verbose=args.v, param_lim=args.plim,
                                          threads=args.threads)
            elif args.alg == 'cut_dqva':
                out = mis.solve_mis_cut_dqva()

            # We can also hot start the optimization to escape local minima
            elif args.alg == 'qaoaHotStart':
                out_results = []
                for i, init_state in enumerate(init_states):
                    out = mis.solve_mis_qaoa(init_state, G, P=args.P, m=args.m,
                                              sim=args.sim, shots=args.shots,
                                              verbose=args.v)
                    out_results.append((i+1, out))

            elif args.alg == 'dqvaHotStart':
                out_results = []
                for i, init_state in enumerate(init_states):
                    out = mis.solve_mis_dqva(init_state, G, P=args.P, m=args.m,
                                              sim=args.sim, shots=args.shots,
                                              verbose=args.v)
                    out_results.append((i+1, out))

            elif args.alg == 'qlsHotStart':
                out_results = []
                for i, init_state in enumerate(init_states):
                    out = mis.solve_mis_qls(init_state, G, P=args.P, m=args.m,
                                            sim=args.sim, shots=args.shots,
                                            verbose=args.v, param_lim=args.plim)
                    out_results.append((i+1, out))

            # Save the results
            if 'HotStart' in args.alg:
                for result in out_results:
                    if args.plim is None:
                        savename = '{}_{}_P{}_{}_init{}_rep{}.pickle'.format(
                                          graphname, args.alg, args.P, args.sim,
                                          result[0], rep)
                    else:
                        savename = '{}_{}_lim{}_{}_init{}_rep{}.pickle'.format(
                                                 graphname, args.alg, args.plim,
                                                 args.sim, result[0], rep)

                    with open(cur_savepath+savename, 'ab') as pf:
                        pickle.dump({'graph':graphfn, 'out':result[1]}, pf)

            else:
                if args.plim is None:
                    savename = '{}_{}_P{}_{}_rep{}.pickle'.format(graphname,
                                                args.alg, args.P, args.sim, rep)
                else:
                    savename = '{}_{}_lim{}_{}_rep{}.pickle'.format(graphname,
                                             args.alg, args.plim, args.sim, rep)

                with open(cur_savepath+savename, 'ab') as pf:
                    pickle.dump({'graph':graphfn, 'out':out}, pf)

if __name__ == '__main__':
    main()

