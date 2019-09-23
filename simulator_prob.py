from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit import BasicAer, execute
import pickle
import glob
import itertools
import copy
import os
import numpy as np

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def simulate_circ(circ, simulator):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circ, backend=backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    outputstate_ordered = [0 for sv in outputstate]
    for i, sv in enumerate(outputstate):
        reverse_i = reverseBits(i,len(circ.qubits))
        outputstate_ordered[reverse_i] = sv
    if simulator == 'sv':
        return outputstate_ordered
    elif simulator == 'prob':
        output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return output_prob
    else:
        raise Exception('Illegal simulator')

def find_cluster_O_rho_qubits(complete_path_map,cluster_idx):
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for q in path[:-1]:
                if q[0] == cluster_idx:
                    O_qubits.append(q)
            for q in path[1:]:
                if q[0] == cluster_idx:
                    rho_qubits.append(q)
    return O_qubits, rho_qubits

def find_all_simulation_combinations(O_qubits, rho_qubits, num_qubits):
    print('Rho qubits:',rho_qubits)
    all_inits = list(itertools.product(init_states,repeat=len(rho_qubits)))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(num_qubits)]
        for i in range(len(init)):
            complete_init[rho_qubits[i][1]] = init[i]
        complete_inits.append(complete_init)
    print('initializations:',complete_inits)

    print('O qubits:',O_qubits)
    all_meas = list(itertools.product(measurement_basis,repeat=len(O_qubits)))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['I' for i in range(num_qubits)]
        for i in range(len(meas)):
            complete_m[O_qubits[i][1]] = meas[i]
        complete_meas.append(complete_m)
    print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    print(len(combinations))


if __name__ == '__main__':
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','minus','plus_i','minus_i']
    dirname = './data'
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))

    [print(x, complete_path_map[x]) for x in complete_path_map]

    cluster_circ_files = [f for f in glob.glob(dirname+'/cluster_*_circ.p')]
    for cluster_idx in range(len(cluster_circ_files)):
        print('cluster %d'%cluster_idx)
        cluster_prob = {}
        cluster_circ = pickle.load(open(('%s/cluster_%d_circ.p'%(dirname,cluster_idx)), 'rb' ))
        print(cluster_circ)
        O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
        find_all_simulation_combinations(O_qubits, rho_qubits, len(cluster_circ.qubits))
        print('-'*100)