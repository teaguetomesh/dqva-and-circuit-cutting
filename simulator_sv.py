from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit import BasicAer, execute
import pickle
import glob
import itertools
import copy
import os

def simulate_circ(circ, simulator='statevector_simulator'):
    backend = BasicAer.get_backend(simulator)
    job = execute(circ, backend=backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    return outputstate

def find_cluster_cut_qubit(complete_path_map,cluster_idx):
    ancilla_indices = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for cut_qubit in path[1::]:
                if cut_qubit[0] == cluster_idx:
                    ancilla_indices.append(cut_qubit[1])
    return ancilla_indices

def init_key(init,ancilla_indices,num_qubits):
    key = [0 for i in range(num_qubits)]
    for i, idx in enumerate(ancilla_indices):
        key[idx] = init[i]
    key = tuple(key)
    return key

if __name__ == '__main__':
    dirname = './data'
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))

    [print(x, complete_path_map[x]) for x in complete_path_map]

    cluster_circ_files = [f for f in glob.glob(dirname+'/cluster_*_circ.p')]
    for cluster_idx in range(len(cluster_circ_files)):
        cluster_sv = {}
        ancilla_indices = find_cluster_cut_qubit(complete_path_map,cluster_idx)
        all_inits = list(itertools.product(range(0,2),repeat=len(ancilla_indices)))
        print('cluster %d, ancillas:'%cluster_idx,ancilla_indices)

        for init in all_inits:
            # print('init conditions:',init)
            cluster_circ = pickle.load(open( ('%s/cluster_%d_circ.p'%(dirname,cluster_idx)), 'rb' ))
            cluster_dag = circuit_to_dag(cluster_circ)
            for idx in range(len(ancilla_indices)):
                ancilla_qubit = cluster_circ.qubits[ancilla_indices[idx]]
                init_cond = init[idx]
                if init_cond == 1:
                    cluster_dag.apply_operation_front(op=XGate(),qargs=[ancilla_qubit],cargs=[])
            cluster_circ = dag_to_circuit(cluster_dag)
            # print(cluster_circ)
            sv = simulate_circ(cluster_circ)
            init = init_key(init,ancilla_indices,len(cluster_circ.qubits))
            print('saved as:',init)
            cluster_sv[init] = sv
        pickle.dump(cluster_sv, open('%s/cluster_%d_sv.p'%(dirname,cluster_idx), 'wb' ))
        print('-'*100)