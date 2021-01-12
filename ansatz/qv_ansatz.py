from qiskit import QuantumCircuit, AncillaRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
from utils.graph_funcs import *
from utils.helper_funcs import *

def apply_mixer(circ, alpha, init_state, G, anc_idx, barriers,
                decompose_toffoli, mixer_order, verbose=0):

    # Apply partial mixers V_i(alpha_i)
    # Randomly permute the order of the mixing unitaries
    if mixer_order is None:
        mixer_order = list(G.nodes)
    if verbose > 0:
        print('Mixer order:', mixer_order)
    for qubit in mixer_order:
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
            mc_toffoli = ControlledGate('mc_toffoli', len(neighbors)+1, [],
                                        num_ctrl_qubits=len(neighbors),
                                        ctrl_state='0'*len(neighbors),
                                        base_gate=XGate())
            circ.append(mc_toffoli, ctrl_qubits + [circ.ancillas[anc_idx]])

        # apply an X rotation controlled by the state of the ancilla qubit
        circ.crx(2*alpha[qubit], circ.ancillas[anc_idx], circ.qubits[qubit])

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

def gen_qv_ansatz(G, P=1, params=[], init_state=None, barriers=1, decompose_toffoli=1,
                  mixer_order=None, verbose=0):

    nq = len(G.nodes)

    # Step 1: Jump Start
    # Run an efficient classical approximation algorithm to warm-start the optimization
    if init_state is None:
        init_state = '0'*nq

    # Step 2: Mixer Initialization
    # Select any one of the initial strings and apply two mixing unitaries separated by the phase separator unitary
    qv_circ = QuantumCircuit(nq, name='q')

    # Add an ancilla qubit, 1 for each subgraph, for implementing the mixer unitaries
    anc_reg = AncillaRegister(1, 'anc')
    qv_circ.add_register(anc_reg)

    #print('Init state:', init_state)
    for qb, bit in enumerate(reversed(init_state)):
        if bit == '1':
            qv_circ.x(qb)
    if barriers > 0:
        qv_circ.barrier()

    # parse the variational parameters
    # The qv ansatz does NOT turn any partial mixers off
    num_nonzero = nq
    assert (len(params) == (num_nonzero + 1) * P),"Incorrect number of parameters!"
    alpha_list = []
    gamma_list = []
    for p in range(P):
        chunk = num_nonzero + 1
        cur_section = params[p*chunk:(p+1)*chunk]
        alpha_list.append(cur_section[:-1])
        gamma_list.append(cur_section[-1])
    if verbose > 0:
        for i in range(P):
            print('alpha_{}: {}'.format(i, alpha_list[i]))
            print('gamma_{}: {}'.format(i, gamma_list[i]))

    # Construct the qv ansatz
    anc_idx = 0
    for alphas, gamma in zip(alpha_list, gamma_list):
        apply_mixer(qv_circ, alphas, init_state, G, anc_idx, barriers,
                    decompose_toffoli, mixer_order, verbose=verbose)
        if barriers > 0:
            qv_circ.barrier()

        apply_phase_separator(qv_circ, gamma, G)
        if barriers > 0:
            qv_circ.barrier()

    if decompose_toffoli > 1:
        #basis_gates = ['x', 'cx', 'barrier', 'crx', 'tdg', 't', 'rz', 'h']
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        qv_circ = pm.run(qv_circ)

    return qv_circ
