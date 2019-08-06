from qiskit import BasicAer, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.tools.visualization import dag_drawer
import numpy as np
import itertools
import copy
import timeit
import pickle
from mpi4py import MPI
import argparse

def simulate_circ(circ, simulator='statevector_simulator'):
    backend = BasicAer.get_backend(simulator)
    job = execute(circ, backend=backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    outputprob = [np.power(abs(x),2) for x in outputstate]
    return outputprob

def simulate_one_instance(s, cut_edge_input_qubits, cut_edge_output_qubits, circ):
    meas_modifications = len(cut_edge_output_qubits)
    circ_copy = copy.deepcopy(circ)
    circ_dag = circuit_to_dag(circ_copy)
    for idx, cut_s in enumerate(s[:meas_modifications]):
        qubit = cut_edge_output_qubits[idx]
        if cut_s == 1 or cut_s == 2 or cut_s == 7 or cut_s == 8:
            continue
        if cut_s == 3 or cut_s == 4:
            circ_dag.apply_operation_back(op=HGate(),qargs=[qubit])
        if cut_s == 5 or cut_s == 6:
            circ_dag.apply_operation_back(op=SdgGate(),qargs=[qubit])
            circ_dag.apply_operation_back(op=HGate(),qargs=[qubit])
    for idx, cut_s in enumerate(s[meas_modifications:]):
        qubit = cut_edge_input_qubits[idx]
        if cut_s == 1 or cut_s == 7:
            continue
        if cut_s == 2 or cut_s == 8:
            circ_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
        if cut_s == 3:
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
        if cut_s == 4:
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
        if cut_s == 5:
            circ_dag.apply_operation_front(op=SGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
        if cut_s == 6:
            circ_dag.apply_operation_front(op=SGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=HGate(),qargs=[qubit],cargs=[])
            circ_dag.apply_operation_front(op=XGate(),qargs=[qubit],cargs=[])
    instance_outputprob = simulate_circ(dag_to_circuit(circ_dag))
    return instance_outputprob

def simulate_cluster_instances(cluster_circ, perms, cut_edge_input_qubits, cut_edge_output_qubits):
    cluster_instances_outputprob = {}
    for s in perms:
        instance_outputprob = simulate_one_instance(s, cut_edge_input_qubits, cut_edge_output_qubits, cluster_circ)
        cluster_instances_outputprob[s] = instance_outputprob
    return cluster_instances_outputprob

def calculate_perms(cluster_circ, cluster_idx, complete_path_map):
    cut_edge_output_qubits = []
    cut_edge_input_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for idx, (circ_idx, path_qubit) in enumerate(path):
                if idx == 0 and circ_idx==cluster_idx:
                    cut_edge_output_qubits.append(path_qubit)
                elif idx == len(path)-1 and circ_idx==cluster_idx:
                    cut_edge_input_qubits.append(path_qubit)
                elif circ_idx==cluster_idx:
                    cut_edge_output_qubits.append(path_qubit)
                    cut_edge_input_qubits.append(path_qubit)
    perms = list(itertools.product(range(1,7),repeat=len(cut_edge_input_qubits)+len(cut_edge_output_qubits)))
    return perms, cut_edge_input_qubits, cut_edge_output_qubits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI simulator.')
    parser.add_argument('--cluster-index', metavar='N', type=int,
                        help='which cluster to run')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    complete_path_map = pickle.load( open( './data/cpm.p', 'rb' ) )
    cluster_circ = pickle.load( open( './data/cluster_%d.p'%args.cluster_index, 'rb' ) )
    perms, cut_edge_input_qubits, cut_edge_output_qubits = calculate_perms(
        cluster_circ, 0, complete_path_map)

    num_workers = size - 1
    count = int(len(perms)/num_workers)
    remainder = len(perms) % num_workers

    if rank == size-1:
        start = MPI.Wtime()
        for i in range(0,size-1):
            state = MPI.Status()
            runtime = comm.recv(source=MPI.ANY_SOURCE,status=state)
            # print('rank %d runtime ='%state.Get_source(), runtime)
        # print('*'*100)
        end = MPI.Wtime()
        print('total runtime = ', end-start)
    elif rank < remainder:
        perms_start = rank * (count + 1)
        perms_stop = perms_start + count + 1
        rank_perms = perms[perms_start:perms_stop]
        # print('rank %d runs %d-%d, total %d instances' % 
        # (rank, perms_start, perms_stop-1, len(rank_perms)))
        
        start = timeit.default_timer()
        cluster_instances_outputprob = simulate_cluster_instances(
            cluster_circ, rank_perms, cut_edge_input_qubits, cut_edge_output_qubits)
        end = timeit.default_timer()

        # print('rank %d sends runtime ='%rank, end-start)
        
        comm.send(end-start, dest=size-1)
    else:
        perms_start = rank * count + remainder
        perms_stop = perms_start + (count - 1) + 1
        rank_perms = perms[perms_start:perms_stop]
        # print('rank %d runs %d-%d, total %d instances' % 
        # (rank, perms_start, perms_stop-1, len(rank_perms)))
        
        start = timeit.default_timer()
        cluster_instances_outputprob = simulate_cluster_instances(
            cluster_circ, rank_perms, cut_edge_input_qubits, cut_edge_output_qubits)
        end = timeit.default_timer()

        # print('rank %d sends runtime ='%rank, end-start)

        comm.send(end-start, dest=size-1)