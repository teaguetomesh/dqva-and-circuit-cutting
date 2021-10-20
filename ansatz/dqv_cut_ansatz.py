"""
05/10/2021 - Teague Tomesh

The functions in the file are used to generate the Dynamic
Quantum Variational Ansatz (DQVA) in a manner that is amenable
to circuit cutting.
"""
from qiskit import QuantumCircuit, AncillaRegister, converters
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
from utils.graph_funcs import *
from utils.helper_funcs import *

def apply_mixer(circ, alpha, init_state, G, barriers,
                decompose_toffoli, mixer_order, subgraph_dict,
                cut_nodes, hot_nodes, verbose=0):
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
        print('APPLYING MIXER UNITARY')
        print('\t\tMixer order:', mixer_order, 'Cut nodes:', cut_nodes, 'Hot nodes:', hot_nodes)

    # Pad the given alpha parameters to account for the zeroed angles
    pad_alpha = [None]*len(init_state)
    next_alpha = 0
    for qubit in mixer_order:
        bit = list(reversed(init_state))[qubit]
        if bit == '1' or next_alpha >= len(alpha) \
           or ( qubit in cut_nodes and qubit not in hot_nodes ):
            continue
        else:
            pad_alpha[qubit] = alpha[next_alpha]
            next_alpha += 1
    if verbose > 0:
        print('\t\tinit_state: {}\n\t\talpha: {}\n\t\tpad_alpha: {}'.format(init_state,
                                                              alpha, pad_alpha))

    cuts = [] # initialize a trivial set of cuts
    # identify the first qubit in the "second" subgraph,  which is used to
    # identify cut locations
    swap_qubit = mixer_order[0]
    for qubit in mixer_order[1:]:
        if subgraph_dict[qubit] != subgraph_dict[swap_qubit]:
            swap_qubit = qubit
            break
    if verbose:
        print('\t\tSwap qubit =', swap_qubit)

    for qubit in mixer_order:

        # identify the location of cuts
        if qubit == swap_qubit and len(hot_nodes) > 0:
            # find all neighbors of the hot nodes
            hot_neighbors = set.union(*[ set(G.neighbors(node)) for node in hot_nodes ])
            # find all cut qubits in the non-hot graph
            adj_cut_qubits = [ circ.qubits[node] for node in hot_neighbors
                               if subgraph_dict[node] != subgraph_dict[hot_nodes[0]] ]
            # cut after all gates on adj_cut_nodes
            cuts = [ ( qubit, num_gates(circ,qubit) ) for qubit in adj_cut_qubits ]
            if verbose:
                print('\t\tcuts:', cuts)

        if pad_alpha[qubit] == None or not G.has_node(qubit):
            # Turn off mixers for qubits which are already 1
            continue

        neighbors = list(G.neighbors(qubit))
        anc_idx = subgraph_dict[qubit]

        if verbose > 0:
            print('\t\tqubit:', qubit, 'num_qubits =', len(circ.qubits),
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

    return cuts

def apply_phase_separator(circ, gamma, G):
    """
    Apply a parameterized Z-rotation to every qubit
    """
    for qb in G.nodes:
        circ.rz(2*gamma, qb)

# determine the number of gates applied to a given qubit in a circuit
def num_gates(circuit, qubit):
    graph = converters.circuit_to_dag(circuit)
    graph.remove_all_ops_named("barrier")
    return sum([ qubit in node.qargs for node in graph.topological_op_nodes() ])

def gen_dqva(G, partition, cut_nodes, hot_nodes, P=1, params=[], init_state=None,
             barriers=1, decompose_toffoli=1, mixer_order=None, verbose=0):

    nq = len(G.nodes)

    if P != 1:
        raise Exception("P != 1 currently unsupported")

    subgraph_dict = None
    subgraphs, _ = get_subgraphs(G, partition)

    # check that all hot nodes are in the same subgraph
    # this assertion fails if there are *no* hot nodes,
    # ... in which case you should not be using ciruit cutting!
    assert len(set([ node in subgraphs[0] for node in hot_nodes ])) == 1

    # identify the subgraph of every node
    subgraph_dict = {}
    for i, subgraph in enumerate(subgraphs):
        for qubit in subgraph:
            subgraph_dict[qubit] = i

    # sort mixers by subgraph, with the "hot subgraph" first
    if mixer_order is None:
        mixer_order = list(G.nodes)

    hot_subgraph = subgraph_dict[hot_nodes[0]]
    def _node_in_hot_graph(node):
        return subgraph_dict[node] == hot_subgraph
    new_mixer_order = sorted(mixer_order, key=_node_in_hot_graph, reverse=True)
    if new_mixer_order != mixer_order:
        print(f"WARNING: mixer order changed from {mixer_order} to {new_mixer_order} to respect subgraph ordering")
        mixer_order = new_mixer_order

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
    num_nonzero = nq - hamming_weight(init_state)
    # WARNING: this assertion is not performed for cutting because we are too lazy
    #          to figure out how many parameters there should actually be
    #assert (len(params) == (nq + 1) * P), "Incorrect number of parameters!"
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
                            cut_nodes, hot_nodes, verbose=verbose)

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

    if verbose > 0:
        print('\tIn gen_dqva: Outside loop, cuts =', cuts)
        print('Ansatz:')
        print(dqva_circ.draw(fold=800))

    if decompose_toffoli > 1:
        #basis_gates = ['x', 'cx', 'barrier', 'crx', 'tdg', 't', 'rz', 'h']
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        dqva_circ = pm.run(dqva_circ)

    # push cuts forward past single-qubit gates
    # to (possibly) get rid of some trivial single-qubit fragments
    circ_graph = converters.circuit_to_dag(dqva_circ)
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
                    if cut_loc not in [ 0, num_gates(dqva_circ,qubit) ] ]
    if verbose > 0:
        print('\tIn gen_dqva: fixed cuts:', fixed_cuts)

    return dqva_circ, fixed_cuts, mixer_order
