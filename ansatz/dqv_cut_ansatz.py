from utils.graph_funcs import *
from utils.helper_funcs import *

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

def gen_cut_dqva(G, partition, params=[], init_state=None, barriers=1, cut=False,
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
    if verbose > 0:
        print('alpha_1:', alpha_1)
        print('gamma_1:', gamma_1)
        print('alpha_2:', alpha_2)

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
        # in subsequent applications of the mixer unitary, all hot nodes
        # should be turned cold
        hot_nodes = []
        apply_mixer(dqv_circ, alpha_2, init_state, subgraph, anc_idx, cutedges,
                    barriers, decompose_toffoli, mixer_order, hot_nodes, verbose=verbose)

    if decompose_toffoli > 1:
        #basis_gates = ['x', 'cx', 'barrier', 'crx', 'tdg', 't', 'rz', 'h']
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        dqv_circ = pm.run(dqv_circ)

    return dqv_circ

