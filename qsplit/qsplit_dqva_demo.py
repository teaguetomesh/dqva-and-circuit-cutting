#!/usr/bin/env python3

import sys, glob, itertools
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
import qiskit

import utils.graph_funcs as graph_funcs
import utils.helper_funcs as helper_funcs

import qsplit_circuit_cutter as qcc
import qsplit_mlrecon_methods as qmm
import qsplit_dqva_methods as qdm

show_graphs = len(sys.argv) > 1

max_cuts = 1
mixing_layers = 2
shots = 500000

verbosity = 1
barriers = 2
decompose_toffoli = 1 # values != 1 currently not supported

# cutoff for probability distributions
dist_cutoff = 1e-4

##########################################################################################

# pick a graph
test_graphs = glob.glob("benchmark_graphs/N8_p30_graphs/*")
test_graph = np.random.choice(test_graphs)
graph = graph_funcs.graph_from_file(test_graph)
qubit_num = graph.number_of_nodes()

# bisect the graph
partition = kernighan_lin_bisection(graph)
subgraphs, cut_edges = graph_funcs.get_subgraphs(graph, partition)

# identify nodes incident to a cut (cut_nodes), as well as their complement (uncut_nodes)
# choose "hot nodes": nodes incident to a cut to which we will nonetheless
#   apply a partial mixer in the first mixing layer
cut_nodes, hot_nodes = qdm.choose_nodes(graph, subgraphs, cut_edges, max_cuts)
uncut_nodes = list(set(graph.nodes).difference(set(cut_nodes)))

# set the initial state
init_state = "0" * qubit_num

# set variational parameters
# todo: don't apply phase separators in the last mixing layer
uncut_nonzero = len([n for n in uncut_nodes if init_state[n] != "1"])
num_params = mixing_layers * (uncut_nonzero + 1) + len(hot_nodes)
params = list(range(1, num_params + 1))

# choose a random mixing order
mixer_order = list(range(qubit_num))
np.random.shuffle(mixer_order)

##########################################################################################
# generate and cut a circuit

circuit, cuts = qdm.gen_cut_dqva(graph, partition, cut_nodes, mixing_layers=mixing_layers,
                                 params=params, init_state=init_state, barriers=barriers,
                                 decompose_toffoli=decompose_toffoli, mixer_order=mixer_order,
                                 hot_nodes=hot_nodes, verbose=verbosity)
print("circuit:")
print(circuit)
print("cuts:")
print(cuts)
print()

fragments, wire_path_map = qcc.cut_circuit(circuit, cuts)
if verbosity > 0:
    for idx, frag in enumerate(fragments):
        print("fragment:",idx)
        print(frag.draw(fold=120))
        print()
    print("wire path map:")
    for key, val in wire_path_map.items():
        print(key, "-->", val)
    print()

if show_graphs:
    view_partition(kl_bisection, graph)
    plt.show()

##########################################################################################
# compute true circuit output, and circuit output acquried via circuit cutting

def chop_dist(dist):
    return { key : val for key, val in dist.items() if val > dist_cutoff}

# get the actual state / probability distribution for the full circuit
bit_num = qubit_num + len(subgraphs) # include ancilla qubits
all_bits = [ "".join(bits) for bits in itertools.product(["0","1"], repeat = bit_num) ]
actual_state = qmm.get_statevector(circuit)
actual_dist = { "".join(bits) : abs(amp)**2
                for bits, amp in zip(all_bits, actual_state)
                if amp != 0 }

if verbosity > 0:
    print("actual distribution:")
    print(chop_dist(actual_dist))
    print()

# simulate fragments, build fragment models, and recombine fragment models
frag_shots = shots // qmm.fragment_variants(wire_path_map)
frag_data = qmm.collect_fragment_data(fragments, wire_path_map, shots = frag_shots,
                                      tomography_backend = "qasm_simulator")
direct_models = qmm.direct_fragment_model(frag_data)
likely_models = qmm.maximum_likelihood_model(direct_models)
recombined_dist = qmm.recombine_fragment_models(likely_models, wire_path_map)

if verbosity > 0:
    print("recombined distribution:")
    print(chop_dist(recombined_dist))
    print()

recombined_fidelity = qdm.fidelity(recombined_dist, actual_dist)
print("fidelity:",recombined_fidelity)
