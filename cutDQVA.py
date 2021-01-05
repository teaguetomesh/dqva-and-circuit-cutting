import time
import pickle
import numpy as np
import networkx as nx
import queue
import glob
import matplotlib.pyplot as plt

from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
import itertools
from scipy.optimize import minimize

from cutqc.main import CutQC
from qiskit_helper_functions.non_ibmq_functions import read_dict
from qiskit import *
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import XGate
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_histogram
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager

from networkx.algorithms.approximation import independent_set


def get_cut_solution(cutqc, max_subcirc_qubit):
    circname = list(cutqc.circuits.keys())[0]
    subcirc_file = 'cutqc_data/' + circname + '/cc_{}/subcircuits.pckl'.format(max_subcirc_qubit)
    picklefile = open(subcirc_file, 'rb')
    cutsoln = pickle.load(picklefile)
    return cutsoln


def square_graph():
    G = nx.Graph()
    G.add_nodes_from(range(4))
    edge_list = list(range(4)) + [0]
    print(edge_list)
    for i in range(len(edge_list)-1):
        G.add_edge(edge_list[i], edge_list[i+1])
    return G

def bowtie_graph():
    G = nx.Graph()
    G.add_nodes_from(range(7))
    edge_list1 = list(range(4)) + [0]
    edge_list2 = list(range(3,7)) + [3]
    print(edge_list1)
    print(edge_list2)
    for edge_list in [edge_list1, edge_list2]:
        for i in range(len(edge_list)-1):
            G.add_edge(edge_list[i], edge_list[i+1])
    return G

def test_graph(n, p):
    G1 = nx.erdos_renyi_graph(n, p)
    G2 = nx.erdos_renyi_graph(n, p)

    # Make a combined graph using the two subgraphs
    G = nx.Graph()

    # Add nodes and edges from G1
    G.add_nodes_from(G1.nodes)
    G.add_edges_from(G1.edges)

    # Add nodes and edges from G2
    offset = len(G1.nodes)
    
    g2_nodes = [n+offset for n in G2.nodes]
    G.add_nodes_from(g2_nodes)
    
    g2_edges = [(n1+offset, n2+offset) for n1, n2 in G2.edges]
    G.add_edges_from(g2_edges)

    # Connect the two subgraphs
    G.add_edge(list(G1.nodes)[-1], list(G2.nodes)[0]+offset)
    
    return G

def ring_graph(n):
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    edges = [(i, i+1) for i in range(n-1)] + [(n-1, 0)]
    G.add_edges_from(edges)
    return G

def apply_mixer(circ, alpha, init_state, G, anc_idx, cutedges, barriers, decompose_toffoli,
                mixer_order, hot_nodes, verbose=0):
    # Pad the given alpha parameters to account for the zeroed angles
    pad_alpha = []
    next_alpha = 0
    for bit in reversed(init_state):
        if bit == '1':
            pad_alpha.append(None)
        else:
            pad_alpha.append(alpha[next_alpha])
            next_alpha += 1
    #alpha = [a*(1-int(bit)) for a, bit in zip(alpha, reversed(init_state))]

    # apply partial mixers V_i(alpha_i)
    # Randomly permute the order of the mixing unitaries
    if mixer_order is None:
        mixer_order = list(G.nodes)
    for qubit in mixer_order:
        if list(reversed(init_state))[qubit] == '1' or not G.has_node(qubit):
            # Turn off mixers for qubits which are already 1
            continue

        neighbors = list(G.neighbors(qubit))

        if any([qubit in edge for edge in cutedges]):
            if qubit in hot_nodes:
                # This qubit is "hot", add its neighbors in the other subgraph
                # to the list of controls
                other_neighbors = []
                for edge in cutedges:
                    if edge[0] == qubit:
                        other_neighbors.append(edge[1])
                    elif edge[1] == qubit:
                        other_neighbors.append(edge[0])
                neighbors += other_neighbors
            else:
                # This qubit is "cold", its mixer unitary = Identity
                if verbose > 0:
                    print('Qubit {} is cold! Apply Identity mixer'.format(qubit))
                continue

        if verbose > 0:
            print('qubit:', qubit, 'num_qubits =', len(circ.qubits), 'neighbors:', neighbors)

        # construct a multi-controlled Toffoli gate, with open-controls on q's neighbors
        # Qiskit has bugs when attempting to simulate custom controlled gates.
        # Instead, wrap a regular toffoli with X-gates
        ctrl_qubits = [circ.qubits[i] for i in neighbors] 
        if decompose_toffoli > 0:
            # apply the multi-controlled Toffoli, targetting the ancilla qubit
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
            circ.mcx(ctrl_qubits, circ.ancillas[anc_idx])
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
        else:
            mc_toffoli = ControlledGate('mc_toffoli', len(neighbors)+1, [], num_ctrl_qubits=len(neighbors),
                                        ctrl_state='0'*len(neighbors), base_gate=XGate())
            circ.append(mc_toffoli, ctrl_qubits + [circ.ancillas[anc_idx]])

        # apply an X rotation controlled by the state of the ancilla qubit
        circ.crx(2*pad_alpha[qubit], circ.ancillas[anc_idx], circ.qubits[qubit])

        # apply the same multi-controlled Toffoli to uncompute the ancilla
        if decompose_toffoli > 0:
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
            circ.mcx(ctrl_qubits, circ.ancillas[anc_idx])
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
        else:
            circ.append(mc_toffoli, ctrl_qubits + [circ.ancillas[anc_idx]])

        if barriers > 1:
            circ.barrier()

def apply_phase_separator(circ, gamma, G):
    for qb in G.nodes:
        circ.rz(2*gamma, qb)

def view_partition(partition, G):
    node_colors = []
    for node in G.nodes:
        if node in partition[0]:
            node_colors.append('gold')
        else:
            node_colors.append('lightblue')

    edge_colors = []
    for edge in G.edges:
        if (edge[0] in partition[0] and edge[1] in partition[1]) or \
           (edge[0] in partition[1] and edge[1] in partition[0]):
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    nx.draw_spring(G, with_labels=True, node_color=node_colors, edge_color=edge_colors)

def get_subgraphs(G, partition):
    subgraphs = []
    cut_edges = []
    all_edges = G.edges
    for subgraph_nodes in partition:
        subG = nx.Graph()
        subG.add_nodes_from(subgraph_nodes)

        for v1, v2 in all_edges:
            if v1 in subgraph_nodes and v2 in subgraph_nodes:
                subG.add_edge(v1, v2)
            if v1 in subgraph_nodes and v2 not in subgraph_nodes:
                cut_edges.append((v1, v2))

        subgraphs.append(subG)

    return subgraphs, cut_edges

def strip_ancillas(counts, circ):
    num_anc = len(circ.ancillas)
    new_counts = {}
    for key in counts:
        new_counts[key[num_anc:]] = counts[key]
    return new_counts

def gen_dqva(G, partition, params=[], init_state=None, barriers=1, cut=False,
             decompose_toffoli=1, mixer_order=None, hot_nodes=[], verbose=0):

    nq = len(G.nodes)
    if cut:
        subgraphs, cutedges = get_subgraphs(G, partition)
    else:
        subgraphs = [G]
        cutedges = []

    if verbose > 0:
        print('Current partition:', partition)
        print('subgraphs:', [list(g.nodes) for g in subgraphs])
        print('cutedges:', cutedges)
        # The hot nodes parameter controls which of the nodes on the cut edges we will
        # hit with a mixer unitary. The other nodes on the cut are "cold" and their
        # mixer will be Identity
        print('hot nodes:', hot_nodes)

    # Step 1: Jump Start
    # Run an efficient classical approximation algorithm to warm-start the optimization
    # (For now, we will select the trivial set of bitstrings with Hamming weight equal to 1)
    # Each partition should get its own jump start
    if init_state is None:
        sub_strs = []
        for sg in subgraphs:
            cur_strs = []
            for i in range(len(sg.nodes)):
                bitstr = ['0']*len(sg.nodes)
                bitstr[i] = '1'
                cur_strs.append(''.join(bitstr))
            sub_strs.append(cur_strs)

        I_cl = []
        for combo in itertools.product(*sub_strs):
            I_cl.append(''.join(combo))
        init_state = I_cl[0] # for now, select the first initial bitstr

    # Step 2: Mixer Initialization
    # Select any one of the initial strings and apply two mixing unitaries separated by the phase separator unitary
    dqv_circ = QuantumCircuit(nq, name='q')

    # Add an ancilla qubit, 1 for each subgraph, for implementing the mixer unitaries
    anc_reg = AncillaRegister(len(subgraphs), 'anc')
    dqv_circ.add_register(anc_reg)

    #print('Init state:', init_state)
    for qb, bit in enumerate(reversed(init_state)):
        if bit == '1':
            dqv_circ.x(qb)
    if barriers > 0:
        dqv_circ.barrier()

    # parse the variational parameters
    # NOTE: current implementation is hardcoded to p = 1.5
    num_nonzero = nq - hamming_weight(init_state)
    assert (len(params) == 2 * num_nonzero + 1),"Incorrect number of parameters!"
    alpha_1 = params[:num_nonzero]
    gamma_1 = params[num_nonzero:num_nonzero+1][0]
    alpha_2 = params[num_nonzero+1:]

    for anc_idx, subgraph in enumerate(subgraphs):
        apply_mixer(dqv_circ, alpha_1, init_state, subgraph, anc_idx, cutedges,
                    barriers, decompose_toffoli, mixer_order, hot_nodes, verbose=verbose)
    if barriers > 0:
        dqv_circ.barrier()

    for subgraph in subgraphs:
        apply_phase_separator(dqv_circ, gamma_1, subgraph)
    if barriers > 0:
        dqv_circ.barrier()

    for anc_idx, subgraph in enumerate(subgraphs):
        apply_mixer(dqv_circ, alpha_2, init_state, subgraph, anc_idx, cutedges,
                    barriers, decompose_toffoli, mixer_order, hot_nodes, verbose=verbose)

    if decompose_toffoli > 1:
        #basis_gates = ['x', 'cx', 'barrier', 'crx', 'tdg', 't', 'rz', 'h']
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        dqv_circ = pm.run(dqv_circ)

    return dqv_circ

def is_indset(bitstr, G):
    nodes = list(G.nodes)
    ind_set = []
    for idx, bit in enumerate(reversed(bitstr)):
        if bit == '1':
            cur_neighbors = list(G.neighbors(idx))
            for node in ind_set:
                if node in cur_neighbors:
                    return False
            else:
                ind_set.append(idx)
    return True

def hamming_weight(bitstr):
    return sum([1 for bit in bitstr if bit == '1'])

def gen_binary_str(n, bitstr, ret):
    """
    Generate all binary strings of length n
    """
    if n > 0:
        gen_binary_str(n-1, bitstr + '0', ret)
        gen_binary_str(n-1, bitstr + '1', ret)
    else:
        ret.append(bitstr)
    return ret

def brute_force_search(G):
    num_nodes = len((list(G.nodes)))
    bitstrs = gen_binary_str(num_nodes, '', [])
    best_str, best_hamming_weight = '', 0
    for bitstr in bitstrs:
        if is_indset(bitstr, G) and hamming_weight(bitstr) > best_hamming_weight:
            best_str = bitstr
            best_hamming_weight = hamming_weight(bitstr)
    return best_str, best_hamming_weight

def sim_with_cutting(circ, backend, sim, shots, verbose, cut_options=None):

    cut_start_time = time.time()

    circuit_name = 'dqva_circuit'
    circuits = {circuit_name:circ}
    circuit_cases = []

    if cut_options is None:
        max_subcircuit_qubit = len(circ.qubits) - 1
        num_subcircuits = [2]
        max_cuts = 4
    else:
        max_subcircuit_qubit = cut_options['max_subcircuit_qubit']
        num_subcircuits = cut_options['num_subcircuits']
        max_cuts = cut_options['max_cuts']

    cutqc = CutQC(circuits=circuits, max_subcircuit_qubit=max_subcircuit_qubit,
                  num_subcircuits=num_subcircuits, max_cuts=max_cuts,
                  verbose=verbose)

    cutsoln = get_cut_solution(cutqc, max_subcircuit_qubit)

    cut_end_time = time.time()
    print('\tCut circuit into {} subcircuits with {} qubits in {:.3f}s'.format(
              len(cutsoln['subcircuits']),
              [len(subcirc.qubits) for subcirc in cutsoln['subcircuits']],
              cut_end_time - cut_start_time))

    #print('Complete Path Map:')
    #for key in subcircs['complete_path_map']:
    #    print(key, '->', subcircs['complete_path_map'][key])
    #print('positions:', subcircs['positions'])
    #for i, sc in enumerate(subcircs['subcircuits']):
    #    print('Subcirc', i)
    #    print('\tqubits = {}, gate counts = {}'.format(len(sc.qubits), sc.count_ops()))
        #print(sc.draw(fold=200))

    #circuits[circuit_name] = circuit
    circuit_cases.append('%s|%d'%(circuit_name,max_subcircuit_qubit))

    eval_start_time = time.time()

    cutqc.evaluate(circuit_cases=circuit_cases, eval_mode='sv', num_nodes=1,
                   num_threads=1, early_termination=[1], ibmq=None)

    eval_end_time = time.time()
    print('\tEvaluate subcircuits in {:.3f}s'.format(eval_end_time - eval_start_time))

    pp_start_time = time.time()

    rec_layers = cutqc.post_process(circuit_cases=circuit_cases, eval_mode='sv',
                                    num_nodes=1, num_threads=2,
                                    early_termination=1, qubit_limit=10,
                                    recursion_depth=1)

    pp_end_time = time.time()
    print('\tPost-process in {:.3f}s'.format(pp_end_time - pp_start_time))

    #print('rec_layers:')
    reconstructed_prob = rec_layers[0]
    #print(rec_layers)
    #probs = {'{:0{}b}'.format(i, len(circ.qubits)): rec_layers[i] for i in range(len(rec_layers)) if rec_layers[i] > 0}
    #print('probs:', probs)

    #cutqc.verify(circuit_cases=circuit_cases, early_termination=1, num_threads=2,qubit_limit=10,eval_mode='sv')

    reorder_start_time = time.time()

    dest_folder = './cutqc_data/dqva_circuit/cc_8/sv_1_10_2'
    #print('dest_folder:', dest_folder)
    meta_data = read_dict(filename='%s/meta_data.pckl'%dest_folder)
    dynamic_definition_folders = glob.glob('%s/dynamic_definition_*'%dest_folder)
    recursion_depth = len(dynamic_definition_folders)
    #print('recursion depth =', recursion_depth)

    for recursion_layer in range(recursion_depth):
        #print('Recursion layer %d'%recursion_layer)
        #dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
        #build_output = read_dict(filename='%s/build_output.pckl'%(dynamic_definition_folder))
        #reconstructed_prob = build_output['reconstructed_prob']

        probs = cutqc.reorder(circ, reconstructed_prob,
                      cutsoln['complete_path_map'], cutsoln['subcircuits'],
                      meta_data['dynamic_definition_schedule'][recursion_layer])

    reorder_end_time = time.time()
    print('\tReorder in {:.3f}s'.format(reorder_end_time - reorder_start_time))

    return probs

def cut_dqva(init_state, G, m=4, threshold=1e-5, cutoff=5, sim='statevector', shots=8192, verbose=0):

    kl_bisection = kernighan_lin_bisection(G)
    print('kl bisection:', kl_bisection)
    cut_nodes = []
    for node in kl_bisection[0]:
        for neighbor in G.neighbors(node):
            if neighbor in kl_bisection[1]:
                cut_nodes.extend([node, neighbor])
    cut_nodes = list(set(cut_nodes))
    hotnode = cut_nodes[0]
    print(cut_nodes, hotnode)

    backend = Aer.get_backend(sim+'_simulator')
    cur_permutation = list(np.random.permutation(list(G.nodes)))

    cut_options = {'max_subcircuit_qubit':len(G.nodes)+len(kl_bisection)-1,
                   'num_subcircuits':[2],
                   'max_cuts':4}

    history = []

    def f(params):
        # Generate a circuit
        # Circuit cutting is not required here, but the circuit should be generated using
        # as much info about the cutting as possible
        dqv_circ = gen_dqva(G, kl_bisection, params=params,
                            init_state=cur_init_state, cut=True,
                            mixer_order=cur_permutation, verbose=verbose,
                            decompose_toffoli=2, barriers=0, hot_nodes=[hotnode])

        # Compute the cost function
        # Circuit cutting will need to be used to perform the execution
        start_time = time.time()
        probs = sim_with_cutting(dqv_circ, backend, sim, shots, verbose,
                                 cut_options=cut_options)
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
            out = minimize(f, x0=init_params, method='COBYLA')
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
            #result = execute(dqv_circ, backend=Aer.get_backend('statevector_simulator')).result()
            #statevector = Statevector(result.get_statevector(dqv_circ))
            #counts = strip_ancillas(statevector.probabilities_dict(decimals=5), dqv_circ)
            counts = sim_with_cutting(dqv_circ, backend, sim, shots, verbose,
                                      cut_options=cut_options)

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
                            'cost':opt_cost, 'permutation':cur_permutation, 'topcounts':top_counts,
                            'previnit':prev_init_state}

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

def main():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)])
    print(list(G.edges()))

    out = cut_dqva('0'*len(G.nodes), G, m=4, threshold=1e-5, cutoff=5, sim='qasm', shots=8192, verbose=0)

if __name__ == '__main__':
    main()
