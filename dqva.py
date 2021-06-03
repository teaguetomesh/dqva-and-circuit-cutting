"""
01/30/2021 - Teague Tomesh

This file contains a set of functions for solving the maximum independent set
(MIS) problem on a given graph using a variety of ansatzes.

Each ansatz (qaoa, dqva, qlsa, dqva+cutting) has its own function which
implements the variational algorithm used to find the MIS.
"""
import time, random, queue, copy, itertools
import numpy as np
import networkx as nx

from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from scipy.optimize import minimize

#from cutqc.main import CutQC
import qiskit
from qiskit import *
from qiskit.quantum_info import Statevector

from ansatz import qaoa, dqv_ansatz, qls_ansatz, dqv_cut_ansatz

import qsplit_circuit_cutter as qcc
import qsplit_mlrecon_methods as qmm

from utils.graph_funcs import *
from utils.helper_funcs import *
from utils.cutting_funcs import *


def solve_mis_cut_dqva(init_state, graph, P=1, m=4, threshold=1e-5, cutoff=1,
                       sim='aer', shots=8192, verbose=0, max_cuts=1):
    """
    Find the MIS of G using the dqva and circuit cutting
    """

    # Initialization
    # NOTE: the backend to use is very version dependent. 
    # Qiskit 0.23.2 does not support the newer Aer_simulators that
    # are available in Qiskit 0.26.0.
    # For now, just use the statevector_simulator
    #backend = Aer.get_backend(name='aer_simulator', method='statevector')
    backend = Aer.get_backend('qasm_simulator')


    # Randomly permute the order of the partial mixers
    cur_permutation = list(np.random.permutation(list(graph.nodes)))

    history = []

    # build circuit fragments and stitching data
    def _get_circuit_and_cuts(num_params, init_state, mixer_order):
        params = [qiskit.circuit.Parameter('var_{}'.format(num)) for num in range(num_params)]

        kwargs = dict(params=params, init_state=init_state,
                      mixer_order=mixer_order, decompose_toffoli=1,
                      verbose=0, P=P)

        circuit, cuts = dqv_cut_ansatz.gen_dqva(graph, partition, cut_nodes,
                                                hot_nodes, **kwargs)
        fragments, wire_path_map = qcc.cut_circuit(circuit, cuts)
        if verbose:
            print('Found cut locations:', cuts)
            print('Cut {}-qubit circuit into {} fragments with ({})-qubits'.format(
                                             circuit.num_qubits, len(fragments),
                                             [f.num_qubits for f in fragments]))
        return fragments, wire_path_map, cuts

    # strip a string of non-digit characters
    def _digit_substr(string):
        return "".join(filter(str.isdigit,string))

    # bind numerical values to the parameters of a circuit
    def _bind(circuit, params):
        binding = { circuit_param : params[int(_digit_substr(circuit_param.name))]
                    for circuit_param in circuit.parameters }
        return circuit.bind_parameters(binding)

    # get output (probability distribution) of a circuit
    def _get_circuit_output(params, var_fragments, wire_path_map, frag_shots):
        start_time = time.time()
        fragments = [ _bind(fragment, params) for fragment in var_fragments ]
        recombined_dist = sim_with_cutting(fragments, wire_path_map, frag_shots,
                                           backend, verbose=0)
        end_time = time.time()
        if verbose:
            print('\t\tsim_with_cutting elapsed time: {:.3f}'.format(end_time-start_time))
        return recombined_dist

    # This function will be what scipy.minimize optimizes
    def avg_cost(params, *args):
        # get output probability distribution for the circuit
        start = time.time()
        probs = _get_circuit_output(params, *args)

        # Compute the average Hamming weight.
        # Have to check each string to ensure it is a valid IS because of the
        # noise introduced by the cutting process.
        avg_weight = sum([prob * hamming_weight(bitstr) for bitstr, prob \
                          in probs.items() if is_indset(bitstr, graph)])
        end = time.time()

        if verbose:
            print('\t\t\tTotal time = {:.3f}, avg weight = {:.4f}'.format(
                                                         end-start, avg_weight))

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
        #while True:
        # Try a single iteration for now
        while inner_round == 1:
            if verbose:
                print('Start round {}.{}, Initial state = {}'.format(mixer_round,
                                                   inner_round, cur_init_state))

            # Begin Inner variational loop
            #     - build parameterized fragments and optimize

            # TODO: fix the num_params computation
            num_params = P * (graph.number_of_nodes() + 1)
            init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)

            # Partition the graph and find cut locations to split the circuit
            cut_start_time = time.time()
            # Sometimes the cutter will fail to find any cuts, in which case
            # the code will break down. Loop to prevent this
            fragments = [None]
            counter = 0
            while len(fragments) == 1 or len(found_cuts) == 0:
                counter += 1
                # Kernighan-Lin partitions a graph into two relatively equal subgraphs
                partition = kernighan_lin_bisection(graph)

                subgraphs, cut_edges = get_subgraphs(graph, partition)

                # identify nodes incident to a cut (cut_nodes),
                # and choose "hot nodes": a subset of cut_nodes to which we will
                # apply a partial mixer in the first mixing layer
                cut_nodes, hot_nodes = simple_choose_nodes(graph, subgraphs, cut_edges, max_cuts, init_state)
                uncut_nodes = list(set(graph.nodes).difference(set(cut_nodes)))

                fragments, wire_path_map, found_cuts = _get_circuit_and_cuts(num_params,
                                                cur_init_state, cur_permutation)
                if len(fragments) == 1 or len(found_cuts) == 0:
                    cur_permutation = list(np.random.permutation(list(graph.nodes)))
                if counter > 100:
                    print('Unable to find viable cuts after 100 iterations!')
                    print('Returning current solution:', best_indset)

            frag_shots = shots // qmm.fragment_variants(wire_path_map)
            cut_end_time = time.time()

            if verbose:
                print('kl bisection:', partition)
                print('Hot nodes:', hot_nodes)
                print('\tNum params =', num_params)
                print('\tCurrent Mixer Order:', cur_permutation)
                print('\tSplit circuit into {} subcircuits with {} qubits in {:.3f} s'.format(
                       len(fragments), [len(frag.qubits) for frag in fragments],
                       cut_end_time - cut_start_time))

            args = (fragments, wire_path_map, frag_shots)
            out = minimize(avg_cost, init_params, args=args, method='COBYLA')
            opt_params = out['x']
            opt_cost = out['fun']
            if verbose:
                print('\tOptimal cost:', opt_cost)
                print('\t{} function evaluations'.format(out['nfev']))

            # Get the results of the optimized circuit
            probs = _get_circuit_output(opt_params, *args)

            # Select the top [cutoff] probs
            top_probs = sorted([(key, val) for key, val in probs.items() if val > threshold],
                                key=lambda tup: tup[1], reverse=True)[:cutoff]

            # Check if we have improved the Hamming weight
            best_hamming_weight = hamming_weight(best_indset)
            better_strs = []
            for bitstr, prob in top_probs:
                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, graph) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # Save current results to history
            inner_history = {'mixer_round':mixer_round, 'inner_round':inner_round,
                             'cost':opt_cost, 'init_state':cur_init_state,
                             'mixer_order':copy.copy(cur_permutation),
                             'num_params':num_params,
                             'frag_qubits':[f.num_qubits for f in fragments]}
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
    return best_indset, best_params, best_init_state, best_perm, partition, cut_nodes, hot_nodes, history


def solve_mis_qls(init_state, G, P=1, m=1, mixer_order=None, threshold=1e-5,
                   cutoff=1, sim='aer', shots=8192, verbose=0,
                   param_lim=None, threads=0):
    """
    Find the MIS of G using Quantum Local Search (QLS), this
    ansatz is composed of two types of unitaries: the cost unitary U_C and the
    mixer unitary U_M. The mixer U_M is made up of individual partial mixers
    which are independently parametrized.

    QLS's key feature is the parameter limit which truncates the number of
    partial mixers that are applied at any one time, and its dynamic reuse of
    quantum resources (i.e. the partial mixers for qubits which are in the MIS
    are turned off and applied to other qubits not currently in the set)
    """

    # Initialization
    if sim == 'statevector' or sim == 'qasm':
        backend = Aer.get_backend(sim+'_simulator', max_parallel_threads=threads)
    elif sim == 'aer':
        backend = Aer.get_backend(name='aer_simulator', method='statevector',
                                      max_parallel_threads=threads)
    elif sim == 'cloud':
        raise Exception('NOT YET IMPLEMENTED')
    else:
        raise Exception('Unknown simulator:', sim)

    # Select an ordering for the partial mixers
    if mixer_order == None:
        cur_permutation = list(np.random.permutation(list(G.nodes)))
    else:
        cur_permutation = mixer_order

    history = []

    # This function will be what scipy.minimize optimizes
    def f(params):
        # Generate a circuit
        circ = qls_ansatz.gen_qlsa(G, P=P, params=params,
                     init_state=cur_init_state, barriers=0, decompose_toffoli=1,
                    mixer_order=cur_permutation, verbose=0, param_lim=param_lim)

        if sim == 'qasm' or sim == 'aer':
            circ.measure_all()

        # Compute the cost function
        result = execute(circ, backend=backend, shots=shots).result()
        if sim == 'statevector':
            statevector = Statevector(result.get_statevector(circ))
            probs = strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == 'qasm' or sim == 'aer':
            counts = result.get_counts(circ)
            probs = strip_ancillas({key: val/shots for key, val in counts.items()}, circ)

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        #print('Expectation value:', avg_cost)
        return -avg_cost

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
        while True:
            print('Start round {}.{}, Initial state = {}'.format(mixer_round,
                  inner_round, cur_init_state))

            # Begin Inner variational loop
            num_nonzero = len(G.nodes()) - hamming_weight(cur_init_state)
            if param_lim is None:
                num_params = min(P * (len(G.nodes()) + 1), (P+1) * (num_nonzero + 1))
            else:
                num_params = param_lim
            print('\tNum params =', num_params)
            # Important to start from random initial points
            #init_params = np.zeros(num_params)
            init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)
            print('\tCurrent Mixer Order:', cur_permutation)

            out = minimize(f, x0=init_params, method='COBYLA')

            opt_params = out['x']
            opt_cost = out['fun']
            #print('\tOptimal Parameters:', opt_params)
            print('\tOptimal cost:', opt_cost)

            # Get the results of the optimized circuit
            opt_circ = qls_ansatz.gen_qlsa(G, P=P, params=opt_params,
                               init_state=cur_init_state, barriers=0,
                               decompose_toffoli=1, mixer_order=cur_permutation,
                               verbose=0, param_lim=param_lim)

            if sim == 'qasm' or sim == 'aer':
                opt_circ.measure_all()

            result = execute(opt_circ, backend=backend, shots=shots).result()
            if sim == 'statevector':
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = strip_ancillas(statevector.probabilities_dict(decimals=5), opt_circ)
            elif sim == 'qasm' or sim == 'aer':
                counts = result.get_counts(opt_circ)
                probs = strip_ancillas({key: val/shots for key, val in counts.items()}, opt_circ)

            # Select the top [cutoff] counts
            top_counts = sorted([(key, val) for key, val in probs.items() if val > threshold],
                                key=lambda tup: tup[1], reverse=True)[:cutoff]

            # Check if we have improved the Hamming weight
            best_hamming_weight = hamming_weight(best_indset)
            better_strs = []
            for bitstr, prob in top_counts:
                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, G) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # Save current results to history
            inner_history = {'mixer_round':mixer_round, 'inner_round':inner_round,
                             'cost':opt_cost, 'init_state':cur_init_state,
                             'mixer_order':copy.copy(cur_permutation), 'num_params':num_params}
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

            # Go through another execution of this While loop, with the same
            # mixer order
            inner_round += 1

        # Save the history of the current mixer round
        history.append(mixer_history)

        # Choose a new permutation of the mixer unitaries that have NOT been set to identity
        identity_mixers = [i for i in range(len(cur_init_state)) if list(reversed(cur_init_state))[i] == '1']
        non_identity_mixers = [i for i in range(len(cur_init_state)) if list(reversed(cur_init_state))[i] == '0']
        permutation = np.random.permutation(non_identity_mixers)
        perm_queue = queue.Queue()
        for p in permutation:
            perm_queue.put(p)
        for i, mixer in enumerate(cur_permutation):
            if mixer in identity_mixers:
                continue
            else:
                cur_permutation[i] = perm_queue.get()

    print('\tRETURNING, best hamming weight:', new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm, history


def solve_mis_qaoa(init_state, G, P=1, m=1, mixer_order=None, threshold=1e-5,
                   cutoff=1, sim='aer', shots=8192, verbose=0,
                   threads=0):
    """
    Find the MIS of G using a Quantum Alternating Operator Ansatz (QAOA), the
    structure of the driver and mixer unitaries is the same as that used by
    DQVA and QLS, but each unitary is parameterized by a single angle:

        U_C_P(gamma_P) * U_M_P(beta_P) * ... * U_C_1(gamma_1) * U_M_1(beta_1)|0>
    """

    # Initialization
    if sim == 'statevector' or sim == 'qasm':
        backend = Aer.get_backend(sim+'_simulator', max_parallel_threads=threads)
    elif sim == 'aer':
        backend = Aer.get_backend(name='aer_simulator', method='statevector',
                                      max_parallel_threads=threads)
    elif sim == 'cloud':
        raise Exception('NOT YET IMPLEMENTED!')
    else:
        raise Exception('Unknown simulator:', sim)

    # Select an ordering for the partial mixers
    if mixer_order == None:
        cur_permutation = list(np.random.permutation(list(G.nodes)))
    else:
        cur_permutation = mixer_order

    history = []

    # This function will be what scipy.minimize optimizes
    def f(params):
        # Generate a QAOA circuit
        circ = qaoa.gen_qaoa(G, P, params=params, init_state=cur_init_state,
                             barriers=0, decompose_toffoli=1,
                             mixer_order=cur_permutation, verbose=0)

        if sim == 'qasm' or sim == 'aer':
            circ.measure_all()

        # Compute the cost function
        result = execute(circ, backend=backend, shots=shots).result()
        if sim == 'statevector':
            statevector = Statevector(result.get_statevector(circ))
            probs = strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == 'qasm' or sim == 'aer':
            counts = result.get_counts(circ)
            probs = strip_ancillas({key: val/shots for key, val in counts.items()}, circ)

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        #print('Expectation value:', avg_cost)
        return -avg_cost

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
        # QAOA only uses a single inner round
        # Break out of the While loop after the 1st iteration
        while inner_round < 2:
            print('Start round {}.{}, Initial state = {}'.format(mixer_round,
                inner_round, cur_init_state))

            # Begin Inner variational loop
            num_params = 2 * P
            print('\tNum params =', num_params)
            # Important to start from random initial points
            #init_params = np.zeros(num_params)
            init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)
            print('\tCurrent Mixer Order:', cur_permutation)

            out = minimize(f, x0=init_params, method='COBYLA')

            opt_params = out['x']
            opt_cost = out['fun']
            #print('\tOptimal Parameters:', opt_params)
            print('\tOptimal cost:', opt_cost)

            # Get the results of the optimized circuit
            opt_circ = qaoa.gen_qaoa(G, P, params=opt_params,
                                     init_state=cur_init_state, barriers=0,
                                     decompose_toffoli=1,
                                     mixer_order=cur_permutation,
                                     verbose=0)

            if sim == 'qasm' or sim == 'aer':
                opt_circ.measure_all()

            result = execute(opt_circ, backend=backend, shots=shots).result()
            if sim == 'statevector':
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = strip_ancillas(statevector.probabilities_dict(decimals=5), opt_circ)
            elif sim == 'qasm' or sim == 'aer':
                counts = result.get_counts(opt_circ)
                probs = strip_ancillas({key: val/shots for key, val in counts.items()}, opt_circ)

            # Select the top [cutoff] bitstrings
            top_counts = sorted([(key, val) for key, val in probs.items() if val > threshold],
                                key=lambda tup: tup[1], reverse=True)[:cutoff]

            # Check if we have improved the Hamming weight
            #     NOTE: hamming_weight(W) = 0
            best_hamming_weight = hamming_weight(best_indset)
            better_strs = []
            for bitstr, prob in top_counts:
                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, G) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # Save current results to history
            inner_history = {'mixer_round':mixer_round, 'inner_round':inner_round,
                             'cost':opt_cost, 'init_state':cur_init_state,
                             'mixer_order':copy.copy(cur_permutation), 'num_params':num_params}
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
        cur_permutation = list(np.random.permutation(list(G.nodes)))

    print('\tRETURNING, best hamming weight:', new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm, history


def solve_mis_dqva(init_state, G, P=1, m=1, mixer_order=None, threshold=1e-5,
                   cutoff=1, sim='aer', shots=8192, verbose=0, threads=0):
    """
    Find the MIS of G using the dynamic quantum variational ansatz (DQVA),
    this ansatz has the same structure as QLS but does not include QLS's
    parameter limit
    """

    # Initialization
    if sim == 'statevector' or sim == 'qasm':
        backend = Aer.get_backend(sim+'_simulator', max_parallel_threads=threads)
    elif sim == 'aer':
        backend = Aer.get_backend(name='aer_simulator', method='statevector',
                                      max_parallel_threads=threads)
    elif sim == 'cloud':
        raise Exception('NOT YET IMPLEMENTED!')
    else:
        raise Exception('Unknown simulator:', sim)

    # Select and order for the partial mixers
    if mixer_order == None:
        cur_permutation = list(np.random.permutation(list(G.nodes)))
    else:
        cur_permutation = mixer_order

    history = []

    # This is the function which scipy.minimize will optimize
    def f(params):
        # Generate a QAOA circuit
        circ = dqv_ansatz.gen_dqva(G, P, params=params,
                     init_state=cur_init_state, barriers=0, decompose_toffoli=1,
                     mixer_order=cur_permutation, verbose=0)

        if sim == 'qasm' or sim == 'aer':
            circ.measure_all()

        # Compute the cost function
        result = execute(circ, backend=backend, shots=shots).result()
        if sim == 'statevector':
            statevector = Statevector(result.get_statevector(circ))
            probs = strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == 'qasm' or sim == 'aer':
            counts = result.get_counts(circ)
            probs = strip_ancillas({key: val/shots for key, val in counts.items()}, circ)

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        #print('Expectation value:', avg_cost)
        return -avg_cost

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
        while True:
            print('Start round {}.{}, Initial state = {}'.format(mixer_round,
                inner_round, cur_init_state))

            # Begin Inner variational loop
            #num_params = P * ((len(G.nodes()) - hamming_weight(cur_init_state)) + 1)
            num_params = P * (len(G.nodes()) + 1)
            print('\tNum params =', num_params)
            # Important to start from random initial points
            #init_params = np.zeros(num_params)
            init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)
            print('\tCurrent Mixer Order:', cur_permutation)

            out = minimize(f, x0=init_params, method='COBYLA')

            opt_params = out['x']
            opt_cost = out['fun']
            #print('\tOptimal Parameters:', opt_params)
            print('\tOptimal cost:', opt_cost)

            # Get the results of the optimized circuit
            opt_circ = dqv_ansatz.gen_dqva(G, P, params=opt_params,
                               init_state=cur_init_state, barriers=0,
                               decompose_toffoli=1, mixer_order=cur_permutation,
                               verbose=0)

            if sim == 'qasm' or sim == 'aer':
                opt_circ.measure_all()

            result = execute(opt_circ, backend=backend, shots=shots).result()
            if sim == 'statevector':
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = strip_ancillas(statevector.probabilities_dict(decimals=5), opt_circ)
            elif sim == 'qasm' or sim == 'aer':
                counts = result.get_counts(opt_circ)
                probs = strip_ancillas({key: val/shots for key, val in counts.items()}, opt_circ)

            # Select the top [cutoff] bitstrings
            top_counts = sorted([(key, val) for key, val in probs.items() if val > threshold],
                                key=lambda tup: tup[1], reverse=True)[:cutoff]

            # Check if we have improved the Hamming weight
            best_hamming_weight = hamming_weight(best_indset)
            better_strs = []
            for bitstr, prob in top_counts:
                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, G) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # Save current results to history
            inner_history = {'mixer_round':mixer_round, 'inner_round':inner_round,
                             'cost':opt_cost, 'init_state':cur_init_state,
                             'mixer_order':copy.copy(cur_permutation), 'num_params':num_params}
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
        cur_permutation = list(np.random.permutation(list(G.nodes)))

    print('\tRETURNING, best hamming weight:', new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm, history

def main():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 3), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)])
    print(list(G.edges()))

    base_str = '0'*len(G.nodes)
    all_init_strs = []
    for i in range(len(G.nodes)):
        init_str = list(base_str)
        init_str[i] = '1'
        out = cut_dqva(''.join(init_str), G, m=4, threshold=1e-5, cutoff=1, sim='qasm', shots=8192, verbose=0)
        print('Init string: {}, Best MIS: {}'.format(''.join(init_str), out[0]))
        print()

if __name__ == '__main__':
    main()
