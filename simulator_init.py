from qiskit import BasicAer, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.tools.visualization import dag_drawer
import numpy as np
import itertools
import copy
import pickle
from mpi4py import MPI
import argparse

def simulate_circ(circ, simulator='statevector_simulator'):
    backend = BasicAer.get_backend(simulator)
    job = execute(circ, backend=backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    # outputprob = [np.power(abs(x),2) for x in outputstate]
    return outputstate

def simulate_one_instance(s, circ):
    circ_copy = copy.deepcopy(circ)
    circ_dag = circuit_to_dag(circ_copy)
    for idx, cut_s in enumerate(s):
        qubit = circ.qubits[idx]
        if cut_s == 1:
            # print('Init to 0')
            continue
        elif cut_s == 2:
            # print('Init to 1')
            circ_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
        elif cut_s == 3:
            # print('Init to +')
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
        elif cut_s == 4:
            # print('Init to -')
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
        elif cut_s == 5:
            # print('Init to +i')
            circ_dag.apply_operation_front(op=SGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
        elif cut_s == 6:
            # print('Init to -i')
            circ_dag.apply_operation_front(op=SGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
        else:
            raise Exception ('Illegal cut_s value in initialization, cut_s =', type(cut_s), cut_s)
    instance_meas = simulate_circ(dag_to_circuit(circ_dag))
    return instance_meas

def simulate_cluster_instances(cluster_circ, perms):
    cluster_meas = {}
    for s in perms:
        instance_meas = simulate_one_instance(s, cluster_circ)
        cluster_meas[tuple(s)] = instance_meas
    return cluster_meas

def calculate_init_perms(cluster_idx, cluster_circ, complete_path_map):
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for circ_idx, rho_qubit in path[1:]:
                if circ_idx == cluster_idx:
                    rho_qubit_idx = cluster_circ.qubits.index(rho_qubit)
                    rho_qubits.append(rho_qubit_idx)
    rho_perms = list(itertools.product(range(1,7),repeat=len(rho_qubits)))
    total_qubits = len(cluster_circ.qubits)
    for idx in range(len(rho_perms)):
        rho_perm = [1 for i in range(total_qubits)]
        for itr, rho_qubit in enumerate(rho_qubits):
            rho_perm[rho_qubit] = rho_perms[idx][itr]
        rho_perms[idx] = rho_perm
    return rho_perms, rho_qubits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI simulator.')
    parser.add_argument('--cluster-file', metavar='S', type=str,
                        help='which cluster pickle file to run')
    args = parser.parse_args()

    complete_path_map = pickle.load( open( './data/cpm.p', 'rb' ) )
    cluster_circ = pickle.load( open( args.cluster_file, 'rb' ) )
    cluster_idx = int(args.cluster_file.split('.p')[0].split('_')[1])
    rho_perms, rho_qubits = calculate_init_perms(cluster_idx, cluster_circ, complete_path_map)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1
    count = int(len(rho_perms)/num_workers)
    remainder = len(rho_perms) % num_workers

    if rank == size-1:
        num_qubits = len(cluster_circ.qubits)
        print('Simulating %d qubit cluster circuit with %d input initializations' % (num_qubits, len(rho_qubits)))
        cluster_meas = {}
        for i in range(0,size-1):
            state = MPI.Status()
            worker_result = comm.recv(source=MPI.ANY_SOURCE,status=state)
            cluster_meas.update(worker_result)
        print('*'*100)
        pickle.dump( cluster_meas, open( './data/cluster_%d_measurement_init.p'%cluster_idx, 'wb' ) )
    elif rank < remainder:
        perms_start = rank * (count + 1)
        perms_stop = perms_start + count + 1
        rank_perms = rho_perms[perms_start:perms_stop]
        
        worker_result = simulate_cluster_instances(
            cluster_circ, rank_perms)
        
        comm.send(worker_result, dest=size-1)
    else:
        perms_start = rank * count + remainder
        perms_stop = perms_start + (count - 1) + 1
        rank_perms = rho_perms[perms_start:perms_stop]
        
        worker_result = simulate_cluster_instances(
            cluster_circ, rank_perms)

        comm.send(worker_result, dest=size-1)