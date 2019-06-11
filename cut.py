import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import pickle
from qiskit import *
from qiskit.visualization import *

def f(zz):
    N = 2**16
    y = [np.exp(z-np.exp(z)) / (1-np.exp(-N)) for z in zz]
    return y

def porter_thomas_distribution(circuit, shots):
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots = shots)
    result = job.result()
    counts = result.get_counts()
    # print(counts)
    N = 2 ** 16
    p = [counts[state]/shots for state in counts]
    z = [math.log10(N*pval) for pval in p]

    return z

# Cut the original circuit into 8^K fragments
# K = number of edges cut
def fragmentation(original_circ, cut_coordinates):
    return original_circ

def main():

    circ = pickle.load( open( "circuit.p", "rb" ) )

    original_z = porter_thomas_distribution(circuit=circ, shots=int(1e4))

    xx = np.arange(-30,3.5,0.1)

    plt.figure()
    plt.plot(xx,f(xx), label = 'Theoretical')
    plt.yscale('log')
    plt.title('Porter Thomas')
    plt.hist(original_z, density=True, histtype='step', label='original circuit simulation')
    plt.legend()
    plt.savefig('Porter Thomas.pdf')

if __name__ == '__main__':
	main()