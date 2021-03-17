import numpy as np
from qiskit import QuantumCircuit, AncillaRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
from utils.graph_funcs import *
from utils.helper_funcs import *

def apply_mixer(circ, G, beta, anc_idx, barriers, decompose_toffoli,
                mixer_order, verbose=0):

    # apply mixers U_M(beta)
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

        # Construct a multi-controlled Toffoli gate, with open-controls on q's neighbors
        # Qiskit has bugs when attempting to simulate custom controlled gates.
        # Instead, wrap a regular toffoli with X-gates

        # Apply the multi-controlled Toffoli, targetting the ancilla qubit
        ctrl_qubits = [circ.qubits[i] for i in neighbors]
        if decompose_toffoli > 0:
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
        circ.crx(2*beta, circ.ancillas[anc_idx], circ.qubits[qubit])

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

def gen_qaoa(G, P, params=[], init_state=None, barriers=1, decompose_toffoli=1,
             mixer_order=None, verbose=0):

    nq = len(G.nodes)

    # Step 1: Jump Start
    if init_state is None:
        # for now, select the all zero state
        init_state = '0'*nq

    # Step 2: Mixer Initialization
    qaoa_circ = QuantumCircuit(nq, name='q')

    # Add an ancilla qubit for implementing the mixer unitaries
    anc_reg = AncillaRegister(1, 'anc')
    qaoa_circ.add_register(anc_reg)

    if init_state == 'W':
        # Prepare the |W> initial state
        W_vector = np.zeros(2**nq)
        for i in range(len(W_vector)):
            bitstr = '{:0{}b}'.format(i, nq)
            if hamming_weight(bitstr) == 1:
                W_vector[i] = 1 / np.sqrt(nq)
        qaoa_circ.initialize(W_vector, qaoa_circ.qubits[:-1])
    else:
        for qb, bit in enumerate(reversed(init_state)):
            if bit == '1':
                qaoa_circ.x(qb)

    if barriers > 0:
        qaoa_circ.barrier()

    # parse the variational parameters
    assert (len(params) == 2*P),"Incorrect number of parameters!"
    betas  = [a for i, a in enumerate(params) if i % 2 == 0]
    gammas = [a for i, a in enumerate(params) if i % 2 == 1]
    if verbose > 0:
        print('betas:', betas)
        print('gammas:', gammas)

    for beta, gamma in zip(betas, gammas):
        anc_idx = 0
        apply_mixer(qaoa_circ, G, beta, anc_idx, barriers, decompose_toffoli,
                    mixer_order, verbose=verbose)
        if barriers > 0:
            qaoa_circ.barrier()

        apply_phase_separator(qaoa_circ, gamma, G)
        if barriers > 0:
            qaoa_circ.barrier()

    if decompose_toffoli > 1:
        #basis_gates = ['x', 'cx', 'barrier', 'crx', 'tdg', 't', 'rz', 'h']
        basis_gates = ['x', 'h', 'cx', 'crx', 'rz', 't', 'tdg', 'u1']
        pass_ = Unroller(basis_gates)
        pm = PassManager(pass_)
        qaoa_circ = pm.run(qaoa_circ)

    return qaoa_circ
