"""
05/10/2021 - Teague Tomesh

The functions in the file are used to generate the Dynamic
Quantum Variational Ansatz (DQVA) in a manner that is amenable
to circuit cutting.
"""
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

from qiskit import QuantumCircuit, AncillaRegister, converters
from qiskit.circuit import ControlledGate, Qubit
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
from utils.graph_funcs import *
from utils.helper_funcs import *

def apply_mixer(circ, alpha, init_state, G, barriers,
                decompose_toffoli, mixer_order, subgraph_dict,
                cut_nodes, hot_nodes, subgraphs, verbose=0):
    """
    Apply the mixer unitary U_M(alpha) to circ

    Input
    -----
    circ : QuantumCircuit
        The current ansatz
    alpha : list[float]
        The angle values of the parametrized gates
    init_state : str
        The current initial state for the ansatz, bits which are "1" are hit
        with an X-gate at the beginning of the circuit and their partial mixers
        are turned off. Bitstring is little-endian ordered.
    G : NetworkX Graph
        The graph we want to solve MIS on
    barriers : int
        An integer from 0 to 2, with 0 having no barriers and 2 having the most
    decompose_toffoli : int
        An integer from 0 to 2, selecting 0 with apply custom open-controlled
        toffoli gates to the ansatz. 1 will apply equivalent gates but using
        instead X-gates and regular-controlled toffolis. 2 unrolls these gates
        to basis gates (but not relevant to this function).
        WARNING Qiskit cannot simulate circuits with decompose_toffoli=0
    mixer_order : list[int]
        The order that the partial mixers should be applied in. For a list
        such as [1,2,0,3] qubit 1's mixer is applied, then qubit 2's, and so on
    cut_nodes : list
        List of nodes indicent to a cut
    subgraph_dict : dict
        A dictionary mapping qubit number to subgraph index
    hot_nodes : list
        A list of "hot nodes" incident to a cut, to which are are applying
        mixers
    verbose : int
        0 is least verbose, 2 is most
    """
    # Apply partial mixers V_i(alpha_i)
    if mixer_order is None:
        mixer_order = list(G.nodes)
    if verbose > 0:
        print('Mixer order:', mixer_order, 'Cut nodes:', cut_nodes, 'Hot nodes:', hot_nodes)

    # Pad the given alpha parameters to account for the zeroed angles
    pad_alpha = [None]*len(init_state)
    next_alpha = 0
    for qubit in mixer_order:
        bit = list(reversed(init_state))[qubit]
        #if (bit == '1' and qubit not in hot_nodes) or next_alpha >= len(alpha) \
        #   or ( qubit in cut_nodes and qubit not in hot_nodes ):
        if qubit in cut_nodes and qubit not in hot_nodes:
            # NOTE: We've relaxed the dynamic turning off of partial mixers for qubits in 1
            # because it creates odd cutting situations where none of the partial mixers
            # in a subgraph are present, so the cutting code doesn't apply and cuts, and we
            # end up simulating the full circuit (see the note in find_cuts)
            continue
        else:
            pad_alpha[qubit] = alpha[next_alpha]
            next_alpha += 1
    if verbose > 0:
        print('init_state: {}\nalpha: {}\npad_alpha: {}'.format(init_state,
                                                              alpha, pad_alpha))
    active_qubits = [v is not None for v in pad_alpha]

    # identify the first qubit in each subgraph after the first, which is used to
    # identify cut locations
    swap_qubits = []
    cur_qubit = mixer_order[0]
    for qubit in mixer_order[1:]:
        if subgraph_dict[qubit] != subgraph_dict[cur_qubit]:
            cur_qubit = qubit
            swap_qubits.append(qubit)
    if verbose:
        print('Swap qubits =', swap_qubits)

    cuts = [] # initialize a trivial set of cuts
    for qubit_index, qubit in enumerate(mixer_order):

        if qubit in swap_qubits and len(hot_nodes) > 0:
            # We just switched subgraphs, identify the cut locations
            cut_these_qubits = find_cuts(G, circ, subgraphs, subgraph_dict,
                                     mixer_order, qubit_index, qubit, hot_nodes,
                                     cut_nodes, active_qubits)

            for qb in cut_these_qubits:
                cuts.append((qb, num_gates(circ, qb)))

            if verbose:
                print('\t\tcuts:', cuts)

        if pad_alpha[qubit] == None or not G.has_node(qubit):
            # Turn off mixers for qubits which are already 1
            continue

        neighbors = list(G.neighbors(qubit))
        anc_idx = subgraph_dict[qubit]

        if verbose > 0:
            print('\tqubit:', qubit, 'num_qubits =', len(circ.qubits),
                  'neighbors:', neighbors)

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
            mc_toffoli = ControlledGate('mc_toffoli', len(neighbors)+1, [],
                                        num_ctrl_qubits=len(neighbors),
                                        ctrl_state='0'*len(neighbors),
                                        base_gate=XGate())
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
    else:
        # Upon exiting the for loop, tie the last subgraph together
        if len(hot_nodes) > 0:
            subgraph_idx = subgraph_dict[qubit]
            tie_subgraph_qubits_together(circ, subgraphs[subgraph_idx], subgraph_idx)

    return cuts


def apply_phase_separator(circ, gamma, G):
    """
    Apply a parameterized Z-rotation to every qubit
    """
    for qb in G.nodes:
        circ.rz(2*gamma, qb)


def num_gates(circuit: QuantumCircuit, qubit: Qubit) -> int:
    """
    Determine the number of gates applied to a given qubit in a circuit
    """
    graph = converters.circuit_to_dag(circuit)
    graph.remove_all_ops_named("barrier")
    return sum([ qubit in node.qargs for node in graph.topological_op_nodes() ])


def find_cuts(G: nx.Graph, circ: QuantumCircuit, subgraphs: nx.Graph,
              subgraph_dict: Dict[int, int], mixer_order: List[int],
              qubit_index: int, qubit: int, hot_nodes: List[int],
              cut_nodes: List[int], active_qubits: List[bool]) -> List[Qubit]:
    """
    This function is called during the construction of the partial mixer when
    switching from one subgraph to another. Identify the qubits that need to be
    cut.

        ansatz overview: [prev_subgraph]---(apply cuts here)--->[cur_subgraph]

    If any of the cut nodes used in the previous subgraph appear in any of the
    later subgraphs, then they must be cut now.
    """
    this_subgraph = subgraph_dict[qubit]
    prev_subgraph = subgraph_dict[mixer_order[qubit_index - 1]]
    tie_subgraph_qubits_together(circ, subgraphs[prev_subgraph], prev_subgraph)

    cut_nodes_in_prev_subgraph = []
    for prev_node in subgraphs[prev_subgraph].nodes:
        # Was this node's partial mixer applied?
        if active_qubits[prev_node]:
            for node in list(G.neighbors(prev_node)) + [prev_node]:
                # If this node is a cut node, add it to the list
                if node in cut_nodes:
                    cut_nodes_in_prev_subgraph.append(node)
    cut_nodes_in_prev_subgraph = list(set(cut_nodes_in_prev_subgraph))

    subgraphs_to_come = list(set([subgraph_dict[mixer_order[i]] for i in range(qubit_index, len(mixer_order))]))
    cut_these_qubits = []
    for node in cut_nodes_in_prev_subgraph:
        # Is this node used in any later subgraphs?
        # NOTE: There is a corner case where in later rounds of the algorithm, when the initial state
        # has a high hamming weight, it's possible that a subgraph will have 0 active qubits. In this
        # case no cuts will be applied to separate the circuit (but in reality it could be separated
        # without any cuts).
        for later_subgraph in subgraphs_to_come:
            later_subgraph_nodes = set(subgraphs[later_subgraph].nodes)
            for later_node in later_subgraph_nodes:
                if active_qubits[later_node] and (node in list(G.neighbors(later_node)) + [later_node]):
                    cut_these_qubits.append(circ.qubits[node])

    return list(set(cut_these_qubits))


def tie_subgraph_qubits_together(circ: QuantumCircuit, subgraph: nx.Graph,
                                 anc_idx: int) -> None:
    """
    The circuit cutting can behave oddly if a qubit doesn't have any two-qubit
    gates applied to it. To keep the subcircuits together and corresponding
    to the subgraphs, apply an identity gate to all qubits in the subgraph.
    """
    identity_circ = QuantumCircuit(len(subgraph.nodes)+1, name='Id')
    for i in range(len(subgraph.nodes)+1):
        identity_circ.id(i)

    identity_gate = identity_circ.to_instruction()

    target_qubits = [circ.qubits[node] for node in subgraph.nodes]
    target_qubits.append(circ.ancillas[anc_idx])
    circ.append(identity_gate, target_qubits)


def gen_dqva(G, partition, cut_nodes, hot_nodes, mixer_order, P=1, params=[], init_state=None,
             barriers=1, decompose_toffoli=1, verbose=0):

    nq = len(G.nodes)

    if P != 1:
        raise Exception("P != 1 currently unsupported")

    subgraphs, _ = get_subgraphs(G, partition)

    # identify the subgraph of every node
    subgraph_dict = {}
    for i, subgraph in enumerate(subgraphs):
        for qubit in subgraph:
            subgraph_dict[qubit] = i

    # Step 1: Jump Start
    # Run an efficient classical approximation algorithm to warm-start the optimization
    if init_state is None:
        init_state = '0'*nq

    # Step 2: Mixer Initialization
    # Select any one of the initial strings and apply two mixing unitaries separated by the phase separator unitary
    dqva_circ = QuantumCircuit(nq, name='q')

    # Add an ancilla qubit(s) for implementing the mixer unitaries
    anc_num = len(partition)
    anc_reg = AncillaRegister(anc_num, 'anc')
    dqva_circ.add_register(anc_reg)

    #print('Init state:', init_state)
    for qb, bit in enumerate(reversed(init_state)):
        if bit == '1':
            dqva_circ.x(qb)
    if barriers > 0:
        dqva_circ.barrier()

    # parse the variational parameters
    # The dqva ansatz dynamically turns off partial mixers for qubits in |1>
    # and adds extra mixers to the end of the circuit
    #num_nonzero = nq - hamming_weight(init_state)
    num_nonzero = nq # Hacking this part because is it breaks for num_fragments > 2
    alpha_list = []
    gamma_list = []
    last_idx = 0
    for p in range(P):
        chunk = num_nonzero + 1
        cur_section = params[p*chunk:(p+1)*chunk]
        alpha_list.append(cur_section[:-1])
        gamma_list.append(cur_section[-1])
        last_idx = (p+1)*chunk

    # Add the leftover parameters as extra mixers
    if len(params[last_idx:]) > 0:
        alpha_list.append(params[last_idx:])

    if verbose > 0:
        print('='*30)
        print('ANSATZ CONSTRUCTION')
        print('Graph partition:', partition)
        print('Parameters:')
        for i in range(len(alpha_list)):
            print('\talpha_{}: {}'.format(i, alpha_list[i]))
            if i < len(gamma_list):
                print('\tgamma_{}: {}'.format(i, gamma_list[i]))

    # Construct the dqva ansatz
    #for alphas, gamma in zip(alpha_list, gamma_list):
    for i in range(len(alpha_list)):
        alphas = alpha_list[i]
        _cuts = apply_mixer(dqva_circ, alphas, init_state, G, barriers,
                            decompose_toffoli, mixer_order, subgraph_dict,
                            cut_nodes, hot_nodes, subgraphs, verbose=verbose)

        if barriers > 0:
            dqva_circ.barrier()

        if i < len(gamma_list):
            gamma = gamma_list[i]
            apply_phase_separator(dqva_circ, gamma, G)

            if barriers > 0:
                dqva_circ.barrier()

        # fix set of cuts and eliminate hot nodes after first mixing layer
        if i == 0:
            cuts = _cuts
            hot_nodes = []
            # Breaking after the first layer of the ansatz is applied.
            # Weird things happen to the cutting when partial mixers past the
            # first layer are applied.
            break

    if verbose > 0:
        print('Ansatz:')
        print(dqva_circ.draw(fold=150))

    if decompose_toffoli > 1:
        #basis_gates = ['x', 'cx', 'barrier', 'crx', 'tdg', 't', 'rz', 'h']
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        dqva_circ = pm.run(dqva_circ)

    return dqva_circ, cuts
