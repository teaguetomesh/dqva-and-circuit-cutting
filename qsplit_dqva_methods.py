#!/usr/bin/env python3

import numpy as np
import networkx as nx
import itertools, qiskit

from qiskit.circuit.library.standard_gates import XGate
from qiskit.circuit import ControlledGate

import utils.graph_funcs as graph_funcs
import utils.helper_funcs as helper_funcs

##########################################################################################

# (1) idetify cut_nodes and uncut_nodes (nodes incident to a cut and their complement)
# (2) choose "hot nodes": nodes incident to a graph partition,
#       to which we will nonetheless apply a partial mixer in the first mixing layer
# WARNING: this algorithm has combinatorial complexity:
#   O({ #cut_nodes_in_subgraph \choose #max_cuts })
# I am simply assuming that this complexity won't be a problem for now
# if it becomes a problem when we scale up, we should rethink this algorithm
def choose_nodes(graph, subgraphs, cut_edges, max_cuts):
    cut_nodes = []
    for edge in cut_edges:
        cut_nodes.extend(edge)
    uncut_nodes = list(set(graph.nodes).difference(set(cut_nodes)))

    # collect subgraph data
    subgraph_A, subgraph_B = subgraphs
    cut_nodes_A = [node for node in subgraph_A.nodes if node in cut_nodes]
    cut_nodes_B = [node for node in subgraph_B.nodes if node in cut_nodes]
    subgraph_cut_nodes = [ (subgraph_A, cut_nodes_A), (subgraph_B, cut_nodes_B) ]

    # compute the cost of each choice of hot nodes
    # hot nodes should all be chosen from one subgraph, so loop over subgraph indices
    choice_cost = {}
    for ext_idx in [ 0, 1 ]:
        # ext_graph: subgraph we're "extending" with nodes from the complement graph
        # ext_cut_nodes: cut_nodes in ext_graph
        ext_graph, ext_cut_nodes = subgraph_cut_nodes[ext_idx]

        # adjacent (complement) graph and cut nodes
        adj_graph, adj_cut_nodes = subgraph_cut_nodes[1-ext_idx]

        # determine the number nodes in adj_cut_nodes that we need to "throw out".
        # nodes that are *not* thrown out are attached to ext_graph in the first mixing layer
        num_to_toss = len(adj_cut_nodes) - max_cuts
        num_to_toss = max(num_to_toss,0)

        # determine size of fragments after circuit cutting.
        # if there are several options (of nodes to toss) with the same "cut cost",
        # these fragment sizes are used to choose between those options
        ext_size = ext_graph.number_of_nodes() + len(adj_cut_nodes) - num_to_toss
        complement_size = subgraphs[1-ext_idx].number_of_nodes()
        frag_sizes = tuple(sorted([ ext_size, complement_size ], reverse = True))

        # if we don't need to throw out any nodes,
        # log a choice_cost of 0 and skip the calculation below
        if num_to_toss == 0:
            choice_cost[ext_idx,()] = (0,) + frag_sizes
            continue

        # for some node (in adj_cut_nodes) that we might throw out
        # (i) determine its neighbors in ext_graph
        # (ii) determine the degrees of those neighbors
        # (iii) add up those degrees
        def single_choice_cost(adj_node):
            return sum([ graph.degree[ext_node]
                         for ext_node in graph.neighbors(adj_node)
                         if ext_node in ext_graph ])

        # loop over all combinations of adjacent nodes that we could throw out
        for toss_nodes in itertools.combinations(adj_cut_nodes, num_to_toss):
             _choice_cost = sum([ single_choice_cost(node) for node in toss_nodes ])
             choice_cost[ext_idx,toss_nodes] = (_choice_cost,) + frag_sizes

    # get the index subgraph we're "extending" and the adjacent nodes we're tossing out
    ext_idx, toss_nodes = min(choice_cost, key = choice_cost.get)
    ext_graph, ext_cut_nodes = subgraph_cut_nodes[ext_idx]

    # determine whether a node in ext_graph has any neighbors in toss_nodes
    def _no_tossed_neighbors(ext_node):
        return not any( neighbor in toss_nodes for neighbor in graph.neighbors(ext_node) )

    # hot nodes = those without neighbors that we are tossing out
    hot_nodes = list(filter(_no_tossed_neighbors, ext_cut_nodes))
    return cut_nodes, uncut_nodes, hot_nodes

##########################################################################################

# fidelity between two probability distributions
def fidelity(dist, actual_dist):
    qubit_num = len(list(actual_dist.keys())[0])
    fidelity = sum( np.sqrt(actual_dist[bits] * dist[bits], dtype = complex)
                    for bits in actual_dist.keys()
                    if actual_dist.get(bits) and dist.get(bits) )**2
    return fidelity.real if fidelity.imag == 0 else fidelity

# determine the number of gates applied to a given qubit in a circuit
def num_gates(circuit, qubit):
    graph = qiskit.converters.circuit_to_dag(circuit)
    graph.remove_all_ops_named("barrier")
    return sum([ qubit in node.qargs for node in graph.topological_op_nodes() ])

def apply_mixer(circ, alpha, init_state, G, cut_nodes, cutedges, subgraph_dict,
                barriers, decompose_toffoli, mixer_order, hot_nodes,
                verbose=0):

    # pad the given alpha parameters to account for the zeroed angles
    pad_alpha = [None]*len(init_state)
    next_alpha = 0
    for qubit in mixer_order:
        bit = list(reversed(init_state))[qubit]
        if bit == '1' or next_alpha >= len(alpha) \
           or (qubit in cut_nodes and qubit not in hot_nodes):
            continue
        else:
            pad_alpha[qubit] = alpha[next_alpha]
            next_alpha += 1
    if verbose > 0:
        print('Mixer order:', mixer_order)
        print('init_state: {}, alpha: {}, pad_alpha: {}'.format(init_state,
                                                                alpha, pad_alpha))
        print('Subgraph dict:', subgraph_dict)

    cuts = [] # initialize a trivial set of cuts

    # identify the first qubit in the second subgraph
    # we identify cuts before applying mixers to this qubit
    if hot_nodes:
        swap_qubit = mixer_order[0]
        for qubit in mixer_order[1:]:
            if subgraph_dict[qubit] != subgraph_dict[swap_qubit]:
                swap_qubit = qubit
                break

    # apply partial mixers V_i(alpha_i)
    for qubit in mixer_order:
        # if appropriate, identify the locations of cuts
        if hot_nodes and qubit == swap_qubit:
            # find all neighbors of the hot nodes
            hot_neighbors = set.union(*[ set(G.neighbors(node)) for node in hot_nodes ])
            # find all cut qubits in the non-hot graph
            adj_cut_qubits = [ circ.qubits[node] for node in hot_neighbors
                               if subgraph_dict[node] != subgraph_dict[hot_nodes[0]] ]
            # cut after all gates on adj_cut_nodes
            cuts = [ ( qubit, num_gates(circ,qubit) ) for qubit in adj_cut_qubits ]

        # turn off mixers for qubits which are already 1
        if pad_alpha[qubit] == None or not G.has_node(qubit):
            continue

        neighbors = list(G.neighbors(qubit))
        anc_idx = subgraph_dict[qubit]

        if verbose > 0:
            print('qubit:', qubit, 'num_qubits =', len(circ.qubits), 'neighbors:', neighbors)

        # construct a multi-controlled Toffoli gate, with open-controls on q's neighbors
        # Qiskit has bugs when attempting to simulate custom controlled gates.
        # Instead, wrap a regular toffoli with X-gates
        ctrl_qubits = [circ.qubits[i] for i in neighbors]

        # apply a multi-controlled Toffoli targeting the ancilla qubit
        def _apply_mc_toffoli():
            if decompose_toffoli > 0:
                for ctrl in ctrl_qubits:
                    circ.x(ctrl)
                circ.mcx(ctrl_qubits, circ.ancillas[anc_idx])
                for ctrl in ctrl_qubits:
                    circ.x(ctrl)
            else:
                mc_toffoli = ControlledGate('mc_toffoli', len(neighbors)+1, [],
                                            num_ctrl_qubits=len(neighbors),
                                            ctrl_state='0'*len(neighbors), base_gate=XGate())
                circ.append(mc_toffoli, ctrl_qubits + [circ.ancillas[anc_idx]])

        _apply_mc_toffoli()

        # apply an X rotation controlled by the state of the ancilla qubit
        circ.crx(2*pad_alpha[qubit], circ.ancillas[anc_idx], circ.qubits[qubit])

        _apply_mc_toffoli()

        if barriers > 1:
            circ.barrier()

    return cuts

def apply_phase_separator(circ, gamma, G):
    for qb in G.nodes:
        circ.rz(2*gamma, qb)

def gen_cut_dqva(G, partition, uncut_nodes, mixing_layers=1, params=[], init_state=None,
                 barriers=1, decompose_toffoli=1, mixer_order=None,
                 hot_nodes=[], verbose=0):
    assert nx.is_connected(G), "we do not currently support disconnected graphs!"

    nq = len(G.nodes)
    subgraphs, cutedges = graph_funcs.get_subgraphs(G, partition)

    # check that all hot nodes are in the same subgraph
    # this assertion fails if there are *no* hot nodes,
    # ... in which case you should not be using ciruit cutting!
    assert len(set([ node in subgraphs[0] for node in hot_nodes ])) == 1

    # identify the subgraph of every node
    subgraph_dict = {}
    for ii, subgraph in enumerate(subgraphs):
        for qubit in subgraph:
            subgraph_dict[qubit] = ii

    # sort mixers by subgraph, with the "hot subgraph" first
    if mixer_order is None:
        mixer_order = list(G.nodes)
    if hot_nodes:
        hot_subgraph = subgraph_dict[hot_nodes[0]]
        def _node_in_hot_graph(node):
            return subgraph_dict[node] == hot_subgraph
        new_mixer_order = sorted(mixer_order, key = _node_in_hot_graph, reverse = True)
        if new_mixer_order != mixer_order:
            print(f"WARNING: mixer order changed from {mixer_order} to {new_mixer_order}")
            mixer_order = new_mixer_order

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
        init_state = '0'*nq

    # Step 2: Mixer Initialization
    dqv_circ = qiskit.QuantumCircuit(nq, name='q')

    # Add an ancilla qubit, 1 for each subgraph, for implementing the mixer unitaries
    anc_reg = qiskit.AncillaRegister(len(subgraphs), 'anc')
    dqv_circ.add_register(anc_reg)

    #print('Init state:', init_state)
    for qb, bit in enumerate(reversed(init_state)):
        if bit == '1':
            dqv_circ.x(qb)
    if barriers > 0:
        dqv_circ.barrier()

    # parse the variational parameters
    cut_nodes = [n for n in G.nodes if n not in uncut_nodes]
    uncut_nonzero = len([n for n in uncut_nodes if init_state[n] != '1'])
    num_params = mixing_layers * (uncut_nonzero + 1) + len(hot_nodes)
    assert (len(params) == num_params),"Incorrect number of parameters!"
    alpha_list = []
    gamma_list = []
    for p in range(mixing_layers):
        chunk = uncut_nonzero + 1
        if p == 0:
            chunk += len(hot_nodes)
        cur_selection = params[p*chunk:(p+1)*chunk]
        alpha_list.append(cur_selection[:-1])
        gamma_list.append(cur_selection[-1])

    if verbose > 0:
        for i in range(mixing_layers):
            print('alpha_{}: {}'.format(i, alpha_list[i]))
            print('gamma_{}: {}'.format(i, gamma_list[i]))

    for i, (alphas, gamma) in enumerate(zip(alpha_list, gamma_list)):
        _cuts = apply_mixer(dqv_circ, alphas, init_state, G, cut_nodes, cutedges,
                            subgraph_dict, barriers, decompose_toffoli, mixer_order,
                            hot_nodes, verbose=verbose)
        if i == 0: cuts = _cuts

        if barriers == 1:
            dqv_circ.barrier()

        apply_phase_separator(dqv_circ, gamma, G)

        if barriers == 0:
            dqv_circ.barrier()

        # in every layer of the ansatz after the first,
        # all hot nodes should be turned cold
        if i == 0:
            hot_nodes = []

    if decompose_toffoli > 1:
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        dqv_circ = pm.run(dqv_circ)

    # push cuts forward past single-qubit gates
    # to (possibly) get rid of some trivial single-qubit fragments
    circ_graph = qiskit.converters.circuit_to_dag(dqv_circ)
    circ_graph.remove_all_ops_named("barrier")
    fixed_cuts = []
    for qubit, cut_loc in cuts:
        qubit_gates = 0
        for node in circ_graph.topological_op_nodes():
            if qubit not in node.qargs: continue
            qubit_gates += 1
            if qubit_gates <= cut_loc: continue
            if len(node.qargs) == 1: cut_loc += 1
            else: break
        fixed_cuts.append( (qubit,cut_loc) )

    # remove trivial cuts at the beginning or end of the circuit
    fixed_cuts = [ (qubit,cut_loc) for qubit, cut_loc in fixed_cuts
                   if cut_loc not in [ 0, num_gates(dqv_circ,qubit) ] ]

    return dqv_circ, fixed_cuts
