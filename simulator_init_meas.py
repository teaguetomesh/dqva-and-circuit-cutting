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
    rho_s, O_s = s
    circ_copy = copy.deepcopy(circ)
    circ_dag = circuit_to_dag(circ_copy)
    for idx, cut_s in enumerate(O_s):
        # print('modifying measurement, cut_s =', cut_s)
        qubit = circ.qubits[idx]
        if cut_s == 1 or cut_s == 2:
            # print('Measure in I')
            continue
        elif cut_s == 3 or cut_s == 4:
            # print('Measure in X')
            circ_dag.apply_operation_back(op=HGate(),qargs=[qubit])
        elif cut_s == 5 or cut_s == 6:
            # print('Measure in Y')
            circ_dag.apply_operation_back(op=SdgGate(),qargs=[qubit])
            circ_dag.apply_operation_back(op=HGate(),qargs=[qubit])
        else:
            raise Exception ('Illegal cut_s value in measurement, cut_s =', type(cut_s), cut_s)
    for idx, cut_s in enumerate(rho_s):
        # print('modifying initialization, cut_s =', cut_s)
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
        rho_s, O_s = s
        cluster_meas[(tuple(rho_s),tuple(O_s))] = instance_meas
    return cluster_meas

def calculate_perms(cluster_idx, cluster_circ, complete_path_map):
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for idx, (circ_idx, path_qubit) in enumerate(path):
                if idx == 0 and circ_idx==cluster_idx:
                    path_qubit_idx = cluster_circ.qubits.index(path_qubit)
                    O_qubits.append(path_qubit_idx)
                elif idx == len(path)-1 and circ_idx==cluster_idx:
                    path_qubit_idx = cluster_circ.qubits.index(path_qubit)
                    rho_qubits.append(path_qubit_idx)
                elif circ_idx==cluster_idx:
                    path_qubit_idx = cluster_circ.qubits.index(path_qubit)
                    O_qubits.append(path_qubit_idx)
                    rho_qubits.append(path_qubit_idx)
    rho_perms = list(itertools.product(range(1,7),repeat=len(rho_qubits)))
    O_perms = list(itertools.product(range(1,7),repeat=len(O_qubits)))
    
    total_qubits = len(cluster_circ.qubits)
    for idx in range(len(rho_perms)):
        rho_perm = [1 for i in range(total_qubits)]
        for itr, rho_qubit in enumerate(rho_qubits):
            rho_perm[rho_qubit] = rho_perms[idx][itr]
        rho_perms[idx] = rho_perm
    for idx in range(len(O_perms)):
        O_perm = [1 for i in range(total_qubits)]
        for itr, O_qubit in enumerate(O_qubits):
            O_perm[O_qubit] = O_perms[idx][itr]
        O_perms[idx] = O_perm
    return rho_perms, O_perms, rho_qubits, O_qubits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI simulator.')
    parser.add_argument('--cluster-file', metavar='S', type=str,
                        help='which cluster pickle file to run')
    args = parser.parse_args()

    complete_path_map = pickle.load( open( './data/cpm.p', 'rb' ) )
    cluster_circ = pickle.load( open( args.cluster_file, 'rb' ) )
    cluster_idx = int(args.cluster_file.split('.p')[0].split('_')[1])
    rho_perms, O_perms, rho_qubits, O_qubits = calculate_perms(cluster_idx, cluster_circ, complete_path_map)
    all_perms = list(itertools.product(rho_perms, O_perms))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1
    count = int(len(all_perms)/num_workers)
    remainder = len(all_perms) % num_workers

    if rank == size-1:
        num_qubits = len(cluster_circ.qubits)
        cluster_idx = int(args.cluster_file.split('_')[1].split('.')[0])
        print('Simulating %d qubit cluster circuit with %d input initializations, %d output measurement basis'
        % (num_qubits, len(rho_qubits),len(O_qubits)))
        cluster_meas = {}
        for i in range(0,size-1):
            state = MPI.Status()
            worker_result = comm.recv(source=MPI.ANY_SOURCE,status=state)
            cluster_meas.update(worker_result)
        print('*'*100)
        pickle.dump( cluster_meas, open( './data/cluster_%d_measurement_init_meas.p'%cluster_idx, 'wb' ) )
    elif rank < remainder:
        perms_start = rank * (count + 1)
        perms_stop = perms_start + count + 1
        rank_perms = all_perms[perms_start:perms_stop]
        
        worker_result = simulate_cluster_instances(
            cluster_circ, rank_perms)
        
        comm.send(worker_result, dest=size-1)
    else:
        perms_start = rank * count + remainder
        perms_stop = perms_start + (count - 1) + 1
        rank_perms = all_perms[perms_start:perms_stop]
        
        worker_result = simulate_cluster_instances(
            cluster_circ, rank_perms)

        comm.send(worker_result, dest=size-1)