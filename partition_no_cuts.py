import time, random, queue, copy, itertools
import numpy as np
import networkx as nx

from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from scipy.optimize import minimize

#from cutqc.main import CutQC
import qiskit
from qiskit import *
from qiskit.quantum_info import Statevector

from ansatz import subgraph_dqva

import qsplit_circuit_cutter as qcc
import qsplit_mlrecon_methods as qmm

from utils.graph_funcs import *
from utils.helper_funcs import *
from utils.cutting_funcs import *


def solve_mis_no_cut_dqva(init_state, graph, P=1, m=4, threshold=1e-5, cutoff=1,
                          sim='aer', shots=8192, verbose=0, max_cuts=1):
    """
    Find the MIS of G using the dqva and partition but no circuit cutting
    """

    # Initialization
    # NOTE: the backend to use is very version dependent. 
    # Qiskit 0.23.2 does not support the newer Aer_simulators that
    # are available in Qiskit 0.26.0.
    # For now, just use the statevector_simulator
    #backend = Aer.get_backend(name='aer_simulator', method='statevector')
    backend = Aer.get_backend('qasm_simulator')

    # Kernighan-Lin partitions a graph into two relatively equal subgraphs
    partition = kernighan_lin_bisection(graph)
    if verbose:
        print('kl bisection:', partition)

    subgraphs, cut_edges = get_subgraphs(graph, partition)
    cut_nodes = []
    for edge in cut_edges:
        cut_nodes.extend(edge)
    cut_nodes = list(set(cut_nodes))

    # Randomly permute the order of the partial mixers
    cur_permutation = list(np.random.permutation(list(graph.nodes)))

    history = []

    # This function will be what scipy.minimize optimizes
    def avg_cost(params, *args):

        subgraph, cut_nodes, init_state, nodes_to_qubits = args
        circ = subgraph_dqva.gen_dqva(subgraph, cut_nodes, nodes_to_qubits,
                                      params=params, init_state=init_state,
                                      barriers=0, full_mixer_order=cur_permutation,
                                      verbose=0)

        circ.measure_all()

        # get output probability distribution for the circuit
        result = execute(circ, backend=backend, shots=shots).result()
        counts = result.get_counts(circ)
        probs = strip_ancillas({key: val/shots for key, val in counts.items()}, circ)

        # Compute the average Hamming weight.
        avg_weight = sum([prob * hamming_weight(bitstr) for bitstr, prob in probs.items()])

        # we want to maximize avg_weight <--> minimize -avg_weight
        return -avg_weight

    # Begin outer optimization loop
    best_indset = init_state
    best_init_state = init_state
    cur_init_state = init_state
    best_params = None
    best_perm = copy.copy(cur_permutation)

    # Randomly permute the order of mixer unitaries m times
    for mixer_round in range(1, m+1):
        mixer_history = []
        inner_round = 1
        new_hamming_weight = hamming_weight(cur_init_state)

        # Attempt to improve the Hamming weight until no further improvements can be made
        # Try a single iteration for now
        while inner_round == 1:
            if verbose:
                print('Start round {}.{}, Initial state = {}'.format(mixer_round,
                                                   inner_round, cur_init_state))

            # Begin Inner variational loop
            if verbose:
                print('\tCurrent Mixer Order:', cur_permutation)

            subgraph_mis = []
            for sub_idx, subgraph in enumerate(subgraphs):
                # Set the correct parameters for the subgraph dqva
                num_params = P * (subgraph.number_of_nodes() + 1)
                init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)

                # Map between Graph nodes and qubits
                nodes_to_qubits = [n for n in subgraph.nodes]
                print('START SUBGRAPH', sub_idx)

                # Form the corresponding initial state
                rev_init_state = list(reversed(cur_init_state))
                sub_init_state = ''.join([rev_init_state[n] for n in reversed(nodes_to_qubits)])

                if verbose:
                    print('\tNum params =', num_params)

                args = (subgraph, cut_nodes, sub_init_state, nodes_to_qubits)
                out = minimize(avg_cost, init_params, args=args, method='COBYLA')
                opt_params = out['x']
                opt_cost = out['fun']
                if verbose:
                    print('\tOptimal cost:', opt_cost)
                    print('\t{} function evaluations'.format(out['nfev']))

                # Get the results of the optimized circuit
                opt_circ = subgraph_dqva.gen_dqva(subgraph, cut_nodes, nodes_to_qubits,
                                                  params=opt_params, init_state=sub_init_state,
                                                  barriers=0, full_mixer_order=cur_permutation,
                                                  verbose=0)
                opt_circ.measure_all()
                result = execute(opt_circ, backend=backend, shots=shots).result()
                counts = result.get_counts(opt_circ)
                probs = strip_ancillas({key: val/shots for key, val in counts.items()}, opt_circ)

                # Select the top [cutoff] probs
                top_probs = sorted([(key, val) for key, val in probs.items() if val > threshold],
                                key=lambda tup: tup[1], reverse=True)[:cutoff]

                if verbose:
                    print('\tFound MIS: {} with probability {:.4f}'.format(top_probs[0][0], top_probs[0][1]))

                assert (cutoff == 1), 'Cutoff must equal 1'
                subgraph_mis.append((top_probs[0][0], nodes_to_qubits))

            # Put the results for the two subgraphs together
            soln_str = ['0'] * len(graph.nodes) # soln_str is big endian ordered here
            for bitstr, nodes2qubits in subgraph_mis:
                for i, bit in enumerate(reversed(bitstr)):
                    if bit == '1':
                        soln_str[nodes2qubits[i]] = '1'

            # After the appropriate nodes have been flipped, switch soln_str to little endian order
            soln_str = ''.join(reversed(soln_str))
            assert (is_indset(soln_str, graph)), 'The solution string is not a valid MIS!'

            if verbose:
                print('Finished optimization. Subgraph results: {} -> Final solution: {}'.format(subgraph_mis, soln_str))

            # Check if we have improved the Hamming weight
            best_hamming_weight = hamming_weight(best_indset)
            this_hamming_weight = hamming_weight(soln_str)
            better_strs = []
            if this_hamming_weight > best_hamming_weight:
                better_strs.append((soln_str, this_hamming_weight))

            # Save current results to history
            inner_history = {'mixer_round':mixer_round, 'inner_round':inner_round,
                             'cost':opt_cost, 'init_state':cur_init_state,
                             'mixer_order':copy.copy(cur_permutation),
                             'num_params':num_params}
            mixer_history.append(inner_history)

            # If no improvement was made, break and go to next mixer round
            if len(better_strs) == 0:
                print('\tNone of the measured bitstrings had higher Hamming weight than:', best_indset)
                break

            # Otherwise, save the new bitstring and repeat
            best_indset, new_hamming_weight = better_strs[0]
            best_init_state = cur_init_state
            best_params = opt_params
            best_perm = copy.copy(cur_permutation)
            cur_init_state = best_indset
            print('\tFound new independent set: {}, Hamming weight = {}'.format(
                                               best_indset, new_hamming_weight))
            inner_round += 1

        # Save the history of the current mixer round
        history.append(mixer_history)

        # Choose a new permutation of the mixer unitaries
        cur_permutation = list(np.random.permutation(list(graph.nodes)))

    print('\tRETURNING, best hamming weight:', new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm, partition, cut_nodes, history


