"""
01/30/2021 - Teague Tomesh

This file contains a set of functions for solving the maximum independent set
(MIS) problem on a given graph using a variety of ansatzes.

Each ansatz (qaoa, qva, dqva, dqva+cutting) has its own function which
implements the variational algorithm used to find the MIS.
"""
import time, random, queue, copy, itertools
import numpy as np
import networkx as nx

from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from scipy.optimize import minimize

#from cutqc.main import CutQC
from qiskit import *
from qiskit.quantum_info import Statevector

from ansatz import qaoa, qv_ansatz, dqv_ansatz, dqv_cut_ansatz

import qsplit_circuit_cutter as qcc
import qsplit_mlrecon_methods as qmm

from utils.graph_funcs import *
from utils.helper_funcs import *

#def get_cut_solution(cutqc, max_subcirc_qubit):
#    circname = list(cutqc.circuits.keys())[0]
#    subcirc_file = 'cutqc_data/' + circname + '/cc_{}/subcircuits.pckl'.format(max_subcirc_qubit)
#    picklefile = open(subcirc_file, 'rb')
#    cutsoln = pickle.load(picklefile)
#    return cutsoln

def sim_with_cutting(circ, cutqc, mip_model, simulation_backend, shots, G,
                     verbose, mode='direct'):
    """
    A helper function which takes a large quantum circuit as input, cuts it into
    two pieces, evaluates each piece independently, and finally stitches the
    results together to simulate the execution of the larger, original circuit.

    Output:
    probs: dict{bitstring : float}
        Outputs a dictionary containing the simulation results. Keys are the
        bitstrings which were observed and their values are the probability that
        they occurred with.
    """

    cut_start_time = time.time()

    #circuits = {'my_circ':circ}

    #if cut_options is None:
    #    max_subcircuit_qubit = len(circ.qubits) - 1
    #    num_subcircuits = [2]
    #    max_cuts = 4
    #else:
    #    max_subcircuit_qubit = cut_options['max_subcircuit_qubit']
    #    num_subcircuits = cut_options['num_subcircuits']
    #    max_cuts = cut_options['max_cuts']

    #cutqc = CutQC(circuits=circuits, max_subcircuit_qubit=max_subcircuit_qubit,
    #              num_subcircuits=num_subcircuits, max_cuts=max_cuts, verbose=verbose)

    #cutsoln = get_cut_solution(cutqc, max_subcircuit_qubit)
    #cutsoln = cutqc.cut_solns[0]
    #if len(cutsoln) == 0:
    #    raise Exception('Cut solution is empty!')
    subcircs, cpm = cutqc.get_subcircs_from_model(circ, mip_model)
    cut_end_time = time.time()
    print('Split circuit into {} subcircuits with {} qubits in {:.3f} s'.format(
               len(subcircs),
               [len(sc.qubits) for sc in subcircs],
               cut_end_time - cut_start_time))

    wpm = {}
    for key in cpm:
        temp = []
        for frag_qubit in cpm[key]:
            temp.append((frag_qubit['subcircuit_idx'], frag_qubit['subcircuit_qubit']))
        wpm[key] = tuple(temp)

    #shots = 999999
    #total_variants = 7
    model_start_time = time.time()
    frag_data = qmm.collect_fragment_data(subcircs, wpm,
                                          shots=shots,
                                          tomography_backend=simulation_backend)

    direct_models = qmm.direct_fragment_model(frag_data)
    model_end_time = time.time()
    recombine_start_time = time.time()
    if mode is 'direct':
        direct_recombined_dist = qmm.recombine_fragment_models(direct_models, wpm)
        dirty_probs = strip_ancillas(qmm.naive_fix(direct_recombined_dist), circ)
    elif mode is 'likely':
        likely_models = qmm.maximum_likelihood_model(direct_models)
        dirty_probs = strip_ancillas(qmm.recombine_fragment_models(likely_models, wpm), circ)
    else:
        raise Exception('Unknown recombination mode:', mode)
    recombine_end_time = time.time()

    clean_start_time = time.time()
    clean_probs = {}
    for bitstr, probability in dirty_probs.items():
        if is_indset(bitstr, G):
            clean_probs[bitstr] = probability

    factor = 1.0 / sum(clean_probs.values())
    probs = {k: v*factor for k, v in clean_probs.items() }
    clean_end_time = time.time()

    print('Model time: {:.3f}, Recombine time: {:.3f}, Clean time {:.3f}'.format(
            model_end_time - model_start_time,
            recombine_end_time - recombine_start_time,
            clean_end_time - clean_start_time))

    return probs

def solve_mis_cut_dqva(init_state, G, m=4, threshold=1e-5, cutoff=1,
                      sim='statevector', shots=8192, verbose=0):
    """
    Find the MIS of G using the dqva and circuit cutting
    """

    # Initialization
    backend = Aer.get_backend(sim+'_simulator')

    # Kernighan-Lin partitions G into two relatively equal subgraphs
    kl_bisection = kernighan_lin_bisection(G)
    print('kl bisection:', kl_bisection)

    # Collect the nodes that have cut edges
    cut_nodes = []
    for node in kl_bisection[0]:
        for neighbor in G.neighbors(node):
            if neighbor in kl_bisection[1]:
                cut_nodes.extend([node, neighbor])
    cut_nodes = list(set(cut_nodes))

    # For now, randomly select a SINGLE node to be the "hot node" - its mixer
    # will be applied across the cut (and so will require circuit cutting).
    # Only using a single hot node now to keep the number of cuts low, but this
    # requirement can be removed in the future.
    hotnode = random.choice(cut_nodes)
    print('Cut nodes and hotnode:', cut_nodes, hotnode)


    # Randomly permute the order of the partial mixers
    cur_permutation = list(np.random.permutation(list(G.nodes)))

    # Options relevant to the circuit cutting code
    cut_options = {'max_subcircuit_qubit':len(G.nodes)+len(kl_bisection)-1,
                   'num_subcircuits':[2],
                   'max_cuts':2}

    history = []

    # This function will be what scipy.minimize optimizes
    def f(params, cutqc, mip_model):
        # Generate a circuit
        # Circuit cutting is not required here, but the circuit should be generated using
        # as much info about the cutting as possible
        dqv_circ = gen_dqva(G, kl_bisection, params=params,
                            init_state=cur_init_state, cut=True,
                            mixer_order=cur_permutation, verbose=verbose,
                            decompose_toffoli=2, barriers=0, hot_nodes=[hotnode])

        # Compute the cost function
        # Circuit cutting will need to be used to perform the execution
        # Time the full cutting+evaluating+reconstruction process
        start_time = time.time()
        probs = sim_with_cutting(dqv_circ, cutqc, mip_model, 'qasm_simulator',
                                 shots, G, verbose)
        end_time = time.time()
        print('Elapsed time: {:.3f}'.format(end_time-start_time))

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        print('Expectation value:', avg_cost)
        return -avg_cost

    # Step 3: Dynamic Ansatz Update
    # Begin outer optimization loop
    best_indset = init_state
    best_init_state = init_state
    cur_init_state = init_state

    # Randomly permute the order of mixer unitaries m times
    for step4_round in range(1, m+1):
        step3_round = 1
        new_hamming_weight = hamming_weight(cur_init_state)
        old_hamming_weight = -1

        # Attempt to improve the Hamming weight until no further improvements can be made
        while new_hamming_weight > old_hamming_weight:
            print('Start round {}.{}, Initial state = {}'.format(step4_round, step3_round, cur_init_state))

            # Inner variational loop
            num_params = 2 * (len(cur_init_state) - hamming_weight(cur_init_state)) + 1
            print('\tNum params =', num_params)
            init_params = np.random.uniform(low=0.0, high=2*np.pi, size=num_params)
            print('\tCurrent Mixer Order:', cur_permutation)
            circuits = {'dqva_circ':gen_dqva(G, kl_bisection, params=init_params,
                                             init_state=cur_init_state, cut=True,
                                             mixer_order=cur_permutation,
                                             verbose=0, decompose_toffoli=2,
                                             barriers=0, hot_nodes=[hotnode])}
            cutqc = CutQC(circuits, cut_options['max_subcircuit_qubit'],
                          cut_options['num_subcircuits'],
                          cut_options['max_cuts'], verbose)
            mip_model = cutqc.get_MIP_model(cut_options['max_subcircuit_qubit'],
                                            cut_options['num_subcircuits'],
                                            cut_options['max_cuts'])
            out = minimize(f, x0=init_params, args=(cutqc, mip_model), method='COBYLA')
            opt_params = out['x']
            opt_cost = out['fun']
            #print('\tOptimal Parameters:', opt_params)
            print('\tOptimal cost:', opt_cost)

            # Get the results of the optimized circuit
            dqv_circ = gen_dqva(G, kl_bisection, params=opt_params,
                                init_state=cur_init_state,
                                mixer_order=cur_permutation, cut=True,
                                verbose=verbose, decompose_toffoli=2,
                                barriers=0, hot_nodes=[hotnode])
            counts = sim_with_cutting(dqv_circ, cutqc, mip_model,
                                      'qasm_simulator', shots, G, verbose)

            #result = execute(dqv_circ, backend=Aer.get_backend('statevector_simulator')).result()
            #statevector = Statevector(result.get_statevector(dqv_circ))
            #counts = strip_ancillas(statevector.probabilities_dict(decimals=5), dqv_circ)

            # Select the top [cutoff] counts
            top_counts = sorted([(key, counts[key]) for key in counts if counts[key] > threshold],
                                key=lambda tup: tup[1], reverse=True)[:cutoff]
            # Check if we have improved the Hamming weight
            old_hamming_weight = hamming_weight(cur_init_state)
            better_strs = []
            for bitstr, prob in top_counts:
                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, G) and this_hamming > old_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)
            prev_init_state = cur_init_state

            # Save current results to history
            temp_history = {'round':'{}.{}'.format(step4_round, step3_round),
                            'cost':opt_cost, 'permutation':cur_permutation,
                            'topcounts':top_counts, 'previnit':prev_init_state}

            # If no improvement was made, break and go to next step4 round
            if len(better_strs) == 0:
                print('\tNone of the measured bitstrings had higher Hamming weight than:', prev_init_state)
                history.append(temp_history)
                break

            # Otherwise, save the new bitstring and check if it is better than all we have seen thus far
            cur_init_state, new_hamming_weight = better_strs[0]
            if new_hamming_weight > hamming_weight(best_indset):
                best_indset = cur_init_state
                best_init_state = prev_init_state
            print('\tFound new independent set: {}, Hamming weight = {}'.format(cur_init_state, new_hamming_weight))
            temp_history['curinit'] = cur_init_state
            history.append(temp_history)
            step3_round += 1

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
    return best_indset, opt_params, best_init_state, kl_bisection, history


def solve_mis_dqva(init_state, G, P=1, m=1, mixer_order=None, threshold=1e-5,
                   cutoff=1, sim='statevector', shots=8192, verbose=0,
                   param_lim=None):
    """
    Find the MIS of G using the Dynamic Quantum Variational Ansatz (DQVA), this
    ansatz is composed of two types of unitaries: the cost unitary U_C and the
    mixer unitary U_M. The mixer U_M is made up of individual partial mixers
    which are independently parametrized.

    DQVA's key feature is the parameter limit which truncates the number of
    partial mixers that are applied at any one time, and its dynamic reuse of
    quantum resources (i.e. the partial mixers for qubits which are in the MIS
    are turned off and applied to other qubits not currently in the set)
    """

    # Initialization
    if sim == 'statevector' or sim == 'qasm':
        backend = Aer.get_backend(sim+'_simulator')
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
        circ = dqv_ansatz.gen_dqva(G, P=P, params=params,
                      init_state=cur_init_state, barriers=0, decompose_toffoli=1,
                      mixer_order=cur_permutation, verbose=0, param_lim=param_lim)

        if sim == 'qasm':
            circ.measure_all()

        # Compute the cost function
        result = execute(circ, backend=backend, shots=shots).result()
        if sim == 'statevector':
            statevector = Statevector(result.get_statevector(circ))
            probs = strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == 'qasm':
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
            init_params = np.zeros(num_params)
            print('\tCurrent Mixer Order:', cur_permutation)

            out = minimize(f, x0=init_params, method='COBYLA')

            opt_params = out['x']
            opt_cost = out['fun']
            #print('\tOptimal Parameters:', opt_params)
            print('\tOptimal cost:', opt_cost)

            # Get the results of the optimized circuit
            opt_circ = dqv_ansatz.gen_dqva(G, P=P, params=opt_params,
                      init_state=cur_init_state, barriers=0,
                      decompose_toffoli=1, mixer_order=cur_permutation,
                      verbose=0, param_lim=param_lim)

            if sim == 'qasm':
                opt_circ.measure_all()

            result = execute(opt_circ, backend=backend, shots=shots).result()
            if sim == 'statevector':
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = strip_ancillas(statevector.probabilities_dict(decimals=5), opt_circ)
            elif sim == 'qasm':
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
                   cutoff=1, sim='statevector', shots=8192, verbose=0):
    """
    Find the MIS of G using a qaoa ansatz, the structure of the driver and mixer
    unitaries is the same as that used by DQVA and QVA, but each unitary is
    parameterized by a single angle:

        U_C_P(gamma_P) * U_M_P(beta_P) * ... * U_C_1(gamma_1) * U_M_1(beta_1)|0>
    """

    # Initialization
    if sim == 'statevector' or sim == 'qasm':
        backend = Aer.get_backend(sim+'_simulator')
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

        if sim == 'qasm':
            circ.measure_all()

        # Compute the cost function
        result = execute(circ, backend=backend, shots=shots).result()
        if sim == 'statevector':
            statevector = Statevector(result.get_statevector(circ))
            probs = strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == 'qasm':
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
            num_params = 2 * P
            print('\tNum params =', num_params)
            init_params = np.zeros(num_params)
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

            if sim == 'qasm':
                opt_circ.measure_all()

            result = execute(opt_circ, backend=backend, shots=shots).result()
            if sim == 'statevector':
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = strip_ancillas(statevector.probabilities_dict(decimals=5), opt_circ)
            elif sim == 'qasm':
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


def solve_mis_qva(init_state, G, P=1, m=1, mixer_order=None, threshold=1e-5,
                   cutoff=1, sim='statevector', shots=8192, verbose=0):
    """
    Find the MIS of G using the quantum variational ansatz (QVA), this ansatz
    has the same structure as DQVA but does not include DQVA's parameter limit
    and dynamic reuse of partial mixers
    """

    # Initialization
    if sim == 'statevector' or sim == 'qasm':
        backend = Aer.get_backend(sim+'_simulator')
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
        circ = qv_ansatz.gen_qv_ansatz(G, P, params=params,
                     init_state=cur_init_state, barriers=0, decompose_toffoli=1,
                     mixer_order=cur_permutation, verbose=0)

        if sim == 'qasm':
            circ.measure_all()

        # Compute the cost function
        result = execute(circ, backend=backend, shots=shots).result()
        if sim == 'statevector':
            statevector = Statevector(result.get_statevector(circ))
            probs = strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == 'qasm':
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
            num_params = P * (len(G.nodes()) + 1)
            print('\tNum params =', num_params)
            init_params = np.zeros(num_params)
            print('\tCurrent Mixer Order:', cur_permutation)

            out = minimize(f, x0=init_params, method='COBYLA')

            opt_params = out['x']
            opt_cost = out['fun']
            #print('\tOptimal Parameters:', opt_params)
            print('\tOptimal cost:', opt_cost)

            # Get the results of the optimized circuit
            opt_circ = qv_ansatz.gen_qv_ansatz(G, P, params=opt_params,
                               init_state=cur_init_state, barriers=0,
                               decompose_toffoli=1, mixer_order=cur_permutation,
                               verbose=0)

            if sim == 'qasm':
                opt_circ.measure_all()

            result = execute(opt_circ, backend=backend, shots=shots).result()
            if sim == 'statevector':
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = strip_ancillas(statevector.probabilities_dict(decimals=5), opt_circ)
            elif sim == 'qasm':
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
