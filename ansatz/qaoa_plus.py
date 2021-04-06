from qiskit import QuantumCircuit, Aer, execute
from utils.helper_funcs import *
from utils.graph_funcs import *
import scipy
import numpy as np


def construct_qaoa_plus(P, G, params, barriers=False, measure=False):
    assert (len(params) == 2*P), "Number of parameters should be 2P"

    nq = len(G.nodes())
    circ = QuantumCircuit(nq, name='q')

    # Initial state
    circ.h(range(nq))

    gammas = [param for i, param in enumerate(params) if i % 2 == 0]
    betas  = [param for i, param in enumerate(params) if i % 2 == 1]
    for i in range(P):
        # Phase Separator Unitary
        for edge in G.edges():
            q_i, q_j = edge
            circ.rz(gammas[i] / 2, [q_i, q_j])
            circ.cx(q_i, q_j)
            circ.rz(-1 * gammas[i] / 2, q_j)
            circ.cx(q_i, q_j)
            if barriers:
                circ.barrier()

        # Mixing Unitary
        for q_i in range(nq):
            circ.rx(-2 * betas[i], q_i)

    if measure:
        circ.measure_all()

    return circ


def expectation_value(counts, G, Lambda):
    total_shots = sum(counts.values())
    energy = 0
    for bitstr, count in counts.items():
        temp_energy = hamming_weight(bitstr)
        for edge in G.edges():
            q_i, q_j = edge
            rev_bitstr = list(reversed(bitstr))
            if rev_bitstr[q_i] == '1' and rev_bitstr[q_j] == '1':
                temp_energy += -1 * Lambda

        energy += count * temp_energy / total_shots

    return energy


def solve_mis(P, G, Lambda):

    backend = Aer.get_backend('qasm_simulator')

    def f(params):
        circ = construct_qaoa_plus(P, G, params, measure=True)

        result = execute(circ, backend=backend, shots=8192).result()
        counts = result.get_counts(circ)

        return -1 * expectation_value(counts, G, Lambda)

    init_params = np.random.uniform(low=0.0, high=2*np.pi, size=2*P)
    out = scipy.optimize.minimize(f, x0=init_params, method='COBYLA')

    return out


def get_ranked_probs(P, G, params, shots=8192):
    circ = construct_qaoa_plus(P, G, params=params, measure=True)
    result = execute(circ, backend=Aer.get_backend('qasm_simulator'), shots=shots).result()
    counts = result.get_counts(circ)

    probs = [(bitstr, counts[bitstr] / shots, is_indset(bitstr, G)) for bitstr in counts.keys()]
    probs = sorted(probs, key=lambda p: p[1], reverse=True)

    return probs


def get_approximation_ratio(out, P, G, shots=8192):
    opt_mis = brute_force_search(G)[1]

    circ = construct_qaoa_plus(P, G, params=out['x'], measure=True)
    result = execute(circ, backend=Aer.get_backend('qasm_simulator'), shots=shots).result()
    counts = result.get_counts(circ)

    # Approximation ratio is computed using ONLY valid independent sets
    # E(gamma, beta) = SUM_bitstrs { (bitstr_counts / total_shots) * hamming_weight(bitstr) } / opt_mis
    numerator = 0
    for bitstr, count in counts.items():
        if is_indset(bitstr, G):
            numerator += count * hamming_weight(bitstr) / shots
    ratio = numerator / opt_mis

    #ratio = sum([count * hamming_weight(bitstr) / shots for bitstr, count in counts.items() \
    #             if is_indset(bitstr, G)]) / opt_mis

    return ratio


def top_strs(counts, G, top=5):
    total_shots = sum(counts.values())
    probs = [(bitstr, counts[bitstr] / total_shots) for bitstr in counts.keys()]
    probs = sorted(probs, key=lambda p: p[1], reverse=True)
    opt_mis = brute_force_search(G)[1]

    for i in range(top):
        ratio = hamming_weight(probs[i][0]) * probs[i][1] / opt_mis
        print('{} ({}) -> {:.4f}%, Ratio = {:.4f}, Is MIS? {}'.format(probs[i][0], hamming_weight(probs[i][0]),
                                                 probs[i][1] * 100, ratio, is_indset(probs[i][0], G)))



