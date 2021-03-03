"""
01/30/2021 - Teague Tomesh

The functions in the file are used to generate the Quantum
Local Search Ansatz (QLSA)
"""
from qiskit import QuantumCircuit, AncillaRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
from utils.graph_funcs import *
from utils.helper_funcs import *

def apply_mixer(circ, alpha, init_state, G, barriers,
                decompose_toffoli, mixer_order, verbose=0):
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
        with an X-gate at the beginning of the circuit and their partial
        mixers are turned off. Bitstring is little-endian ordered.
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
    verbose : int
        0 is least verbose, 2 is most
    """

    # apply partial mixers V_i(alpha_i)
    if mixer_order is None:
        mixer_order = list(G.nodes)
    if verbose > 0:
        print('Mixer order:', mixer_order)

    # Pad the given alpha parameters to account for the zeroed angles
    pad_alpha = [None]*len(init_state)
    next_alpha = 0
    for qubit in mixer_order:
        bit = list(reversed(init_state))[qubit]
        if bit == '1' or next_alpha >= len(alpha):
            continue
        else:
            pad_alpha[qubit] = alpha[next_alpha]
            next_alpha += 1
    if verbose > 0:
        print('init_state: {}, alpha: {}, pad_alpha: {}'.format(init_state,
                                                              alpha, pad_alpha))

    anc_idx = 0
    for qubit in mixer_order:
        if pad_alpha[qubit] == None or not G.has_node(qubit):
            # Turn off mixers for qubits which are already 1
            continue

        neighbors = list(G.neighbors(qubit))

        if verbose > 0:
            print('qubit:', qubit, 'num_qubits =', len(circ.qubits),
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
    """
    Apply a parameterized Z-rotation to every qubit
    """
    for qb in G.nodes:
        circ.rz(2*gamma, qb)

def gen_qlsa(G, P=1, params=[], init_state=None, barriers=1, decompose_toffoli=1,
            mixer_order=None, verbose=0, param_lim=None):
    """
    Generate and return the Quantum Local Search Ansatz (QLSA)

    Input
    -----
    G : NetworkX Graph
        The graph we want to solve MIS on
    P : int
        Controls the number of unitary layers to apply. If param_lim is an
        integer this P value will be ignored
    params : list[float]
        The angles of the parameterized gates
    init_state : str
        The initial state
    barriers : int
        Integer from 0 (no barriers) to 2 (most barriers)
    decompose_toffoli : int
        An integer from 0 to 2, selecting 0 with apply custom open-controlled
        toffoli gates to the ansatz. 1 will apply equivalent gates but using
        instead X-gates and regular-controlled toffolis. 2 unrolls these gates
        to basis gates (but not relevant to this function).
        WARNING Qiskit cannot simulate circuits with decompose_toffoli=0
    mixer_order : list[int]
        The order that the partial mixers should be applied in. For a list
        such as [1,2,0,3] qubit 1's mixer is applied, then qubit 2's, and so on
    verbose : int
        0 is least verbose, 2 is most
    param_lim : int
        Controls the number of parameters allowed in the qls. If this is set,
        then the length of params should match and the P value will also be
        ignored

    Output
    ------
    QuantumCircuit
    """

    nq = len(G.nodes)

    # Step 1: Jump Start
    # Run an efficient classical approximation algorithm to warm-start the optimization
    if init_state is None:
        init_state = '0'*nq

    # Step 2: Mixer Initialization
    qls_circ = QuantumCircuit(nq, name='q')

    # Add an ancilla qubit for implementing the mixer unitaries
    anc_reg = AncillaRegister(1, 'anc')
    qls_circ.add_register(anc_reg)

    #print('Init state:', init_state)
    for qb, bit in enumerate(reversed(init_state)):
        if bit == '1':
            qls_circ.x(qb)
    if barriers > 0:
        qls_circ.barrier()

    # check the number of variational parameters
    num_nonzero = nq - hamming_weight(init_state)
    if param_lim is None:
        num_params = min(P * (nq + 1), (P+1) * (num_nonzero + 1))
    else:
        num_params = param_lim
    assert (len(params) == num_params),"Incorrect number of parameters!"

    # parse the given parameter list into alphas (for the mixers) and
    # gammas (for the drivers)
    alpha_list = []
    gamma_list = []
    param_index = 0
    while param_index < len(params):
        if param_index == 0:
            gamma_list.append(params[param_index])
            param_index += 1
            need_new_driver = False
        elif num_params - param_index >= num_nonzero:
            alpha_list.append(params[param_index:param_index+num_nonzero])
            param_index += num_nonzero
            if param_index < len(params) and need_new_driver:
                gamma_list.append(params[param_index])
                param_index += 1
            need_new_driver = True
        elif num_params - param_index < num_nonzero:
            alpha_list.append(params[param_index:])
            param_index += len(params[param_index:])

    if verbose > 0:
        for i in range(len(alpha_list)):
            print('alpha_{}: {}'.format(i, alpha_list[i]))
            if i < len(gamma_list):
                print('gamma_{}: {}'.format(i, gamma_list[i]))

    # Apply alternating layers of mixer and driver unitaries
    for i in range(len(alpha_list)):
        alphas = alpha_list[i]
        apply_mixer(qls_circ, alphas, init_state, G, barriers,
                    decompose_toffoli, mixer_order, verbose=verbose)

        if barriers == 1:
            qls_circ.barrier()

        if i < len(gamma_list):
            gamma = gamma_list[i]
            apply_phase_separator(qls_circ, gamma, G)

            if barriers == 1:
                qls_circ.barrier()

    if decompose_toffoli > 1:
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        qls_circ = pm.run(qls_circ)

    return qls_circ

