import glob
import dqva
import partition_no_cuts
import pickle
import networkx as nx
from utils.graph_funcs import graph_from_file


cutting = False

#all_graphs = glob.glob("benchmark_graphs/N12_p20_graphs/*")
gfile = 'benchmark_graphs/N26_3regular_cutting_graph.txt'
G = graph_from_file(gfile)
#gfile = '3-regular'
#G = nx.generators.random_graphs.random_regular_graph(3, 26)
print('Generated graph:', gfile, 'with {} nodes'.format(G.number_of_nodes()))

init_state = '0'*G.number_of_nodes()
full_history = []
for rounds in range(4):
    print('-------------- ROUND {} BEGIN --------------\n\n'.format(rounds+1))
    if cutting:
        out = dqva.solve_mis_cut_dqva(init_state, G, m=1, verbose=1,
                                      shots=50000, max_cuts=1)
    else:
        out = partition_no_cuts.solve_mis_no_cut_dqva(init_state, G, m=1,
                                                      shots=20000, verbose=1)
    init_state = out[0]
    full_history.append(out)

with open('benchmark_results/dqva_and_cutting_0cuts.pickle', 'wb') as pf:
    pickle.dump((G, full_history), pf)
