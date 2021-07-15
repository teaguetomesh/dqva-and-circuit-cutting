import glob
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from qiskit import *
from qiskit.quantum_info import Statevector

import mis

from utils.graph_funcs import *
from utils.helper_funcs import *

from ansatz import qaoa, qv_ansatz, dqv_ansatz, dqv_cut_ansatz


def get_data(sample_graphs):

    qaoa_data = {}
    dqva_data = {}

    for graph_name in sample_graphs:
        print(graph_name)
        G = graph_from_file(graph_name)
        nq = len(G.nodes())
        opt_mis = brute_force_search(G)[1]

        init_state = '0'*nq
        mixer_order = list(range(nq))

        graph_key = graph_name.split('/')[-1].strip('.txt')
        graph_qaoa_data = []
        graph_dqva_data = []

        print('\n\nBEGIN QAOA\n\n')

        for P in [1,2,3]:
            output = mis.solve_mis_qaoa(init_state, G, P=P, mixer_order=mixer_order, sim='qasm')
            ap_ratio = hamming_weight(output[0]) / opt_mis
            mixer_count = nq * P
            print('-'*30)
            print('Found approximation ratio = {}, with {} partial_mixers'.format(ap_ratio, mixer_count))
            print('-'*30)
            graph_qaoa_data.append((mixer_count, ap_ratio))

        qaoa_data[graph_key] = graph_qaoa_data

        print('\n\nBEGIN DQVA\n\n')

        for plim in [3, 9, 15, 21]:
            output = mis.solve_mis_dqva(init_state, G, m=5, mixer_order=mixer_order, sim='qasm', param_lim=plim)
            ap_ratio = hamming_weight(output[0]) / opt_mis
            mixer_count = plim - 1
            print('-'*30)
            print('Found approximation ratio = {}, with {} partial_mixers'.format(ap_ratio, mixer_count))
            print('-'*30)
            graph_dqva_data.append((mixer_count, ap_ratio))

        dqva_data[graph_key] = graph_dqva_data

    return qaoa_data, dqva_data

def plot_comparison(qaoa_data, dqva_data, savefig=None, show=True):
    assert(list(qaoa_data.keys()) == list(dqva_data.keys()))

    for graph in qaoa_data.keys():

        fig, ax = plt.subplots(dpi=150)

        for dat, label in zip([qaoa_data, dqva_data], ['QAOA', 'DQVA']):
            xvals = [tup[0] for tup in dat[graph]]
            yvals = [tup[1] for tup in dat[graph]]
            print(label)
            print(xvals)
            print(yvals)

            ax.plot(xvals, yvals, label=label)

        ax.set_title(graph)
        ax.legend()
        ax.set_ylabel('Approximation Ratio')
        ax.set_xlabel('Number of partial mixers')

        if show:
            plt.show()

        if not savefig is None:
            plt.savefig(savefig)

        plt.close()


def main():
    test_graphs = glob.glob('benchmark_graphs/N12_p20_graphs/*')
    test_graphs = sorted(test_graphs, key=lambda g: int(g.split('/')[-1].strip('G.txt')))
    print(len(test_graphs))

    sample_graphs = test_graphs[0:50]
    print(len(sample_graphs))

    qaoa_data, dqva_data = get_data(sample_graphs)

    all_x = []
    all_y = []
    for key, data in qaoa_data.items():
        all_x.append([v[0] for v in data])
        all_y.append([v[1] for v in data])
    all_x = np.mean(all_x, axis=0)
    all_y = np.mean(all_y, axis=0)
    avg_qaoa_data = {'Avg Erdos-Renyi N=12':list(zip(all_x, all_y))}

    all_x = []
    all_y = []
    for key, data in dqva_data.items():
        all_x.append([v[0] for v in data])
        all_y.append([v[1] for v in data])
    all_x = np.mean(all_x, axis=0)
    all_y = np.mean(all_y, axis=0)
    avg_dqva_data = {'Avg Erdos-Renyi N=12':list(zip(all_x, all_y))}

    print(avg_qaoa_data)
    print(avg_dqva_data)

    plot_comparison(avg_qaoa_data, avg_dqva_data,
                    savefig='figures/avg_erdosrenyi_N12_graphs.png', show=False)

if __name__ == '__main__':
    main()
