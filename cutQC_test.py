from helper_functions.non_ibmq_functions import generate_circ
import matplotlib.pyplot as plt
import numpy as np

from cutqc.main import CutQC

if __name__ == '__main__':
    circuit = generate_circ(full_circ_size=5,circuit_type='bv')

    cutqc = CutQC(circuits={'bv':circuit})
    cutqc.cut(max_subcircuit_qubit=3, num_subcircuits=[2], max_cuts=10)
    cutqc.evaluate()