from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
import pickle
import glob
import itertools
import copy
import os
import numpy as np
import progressbar as pb
from time import time
from mpi4py import MPI
import argparse

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def simulate_circ(circ, backend, noisy=False,qasm_info=None):
    if backend == 'statevector_simulator':
        if noisy:
            raise Exception('statevector simulator does not run noisy evaluations')
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_ordered = [0 for sv in outputstate]
        for i, sv in enumerate(outputstate):
            reverse_i = reverseBits(i,len(circ.qubits))
            outputstate_ordered[reverse_i] = sv
        output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return output_prob
        # return outputstate_ordered
    elif backend == 'qasm_simulator':
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        backend = Aer.get_backend('qasm_simulator')
        if noisy:
            noise_model,coupling_map,basis_gates,num_shots,initial_layout = qasm_info
            na_result = execute(experiments=qc,
            backend=backend,
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            shots=num_shots,
            initial_layout=initial_layout).result()
            na_counts = na_result.get_counts(qc)
            na_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
            for state in na_counts:
                reversed_state = reverseBits(int(state,2),len(circ.qubits))
                na_prob[reversed_state] = na_counts[state]/num_shots
            return na_prob
        else:
            _,_,_,num_shots,_ = qasm_info
            job_sim = execute(qc, backend, shots=num_shots)
            result = job_sim.result()
            noiseless_counts = result.get_counts(qc)
            noiseless_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
            for state in noiseless_counts:
                reversed_state = reverseBits(int(state,2),len(circ.qubits))
                noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
            return noiseless_prob
    else:
        raise Exception('Illegal simulator:',backend)

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
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','minus','plus_i','minus_i']
    # print('Rho qubits:',rho_qubits)
    all_inits = list(itertools.product(init_states,repeat=len(rho_qubits)))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(num_qubits)]
        for i in range(len(init)):
            complete_init[rho_qubits[i][1]] = init[i]
        complete_inits.append(complete_init)
    # print('initializations:',complete_inits)

    # print('O qubits:',O_qubits)
    all_meas = list(itertools.product(measurement_basis,repeat=len(O_qubits)))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['I' for i in range(num_qubits)]
        for i in range(len(meas)):
            complete_m[O_qubits[i][1]] = meas[i]
        complete_meas.append(complete_m)
    # print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    return combinations

def evaluate_cluster(complete_path_map, cluster_circ, combinations, backend='statevector_simulator',noisy=False):
    num_shots = int(1e5)
    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    properties = device.properties()
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties)
    basis_gates = noise_model.basis_gates

    cluster_prob = {}
    for _, combination in enumerate(combinations):
        cluster_dag = circuit_to_dag(cluster_circ)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = cluster_circ.qubits[i]
            if x == 'zero':
                continue
            elif x == 'one':
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal initialization : ',x)
        for i,x in enumerate(meas):
            q = cluster_circ.qubits[i]
            if x == 'I':
                continue
            elif x == 'X':
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                cluster_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        cluster_circ_inst = dag_to_circuit(cluster_dag)
        # print(inits, meas)
        # print(cluster_circ_inst)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(cluster_dag)
        initial_layout = noise_mapper.property_set['layout']
        qasm_info = [noise_model,coupling_map,basis_gates,num_shots,initial_layout]
        cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,
        backend=backend,
        noisy=noisy,qasm_info=qasm_info)
        cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
    return cluster_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--cluster-idx', metavar='N', type=int,help='which cluster pickle file to run')
    parser.add_argument('--backend', metavar='S', type=str,help='which Qiskit backend')
    parser.add_argument('--noisy', action='store_true',help='noisy evaluation?')
    parser.add_argument('--dirname', metavar='S', type=str,default='./data',help='which directory?')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    dirname = args.dirname
    clusters, complete_path_map, qasm_info = pickle.load( open( '%s/evaluator_input.p'%dirname, 'rb' ) )

    cluster_circ = clusters[args.cluster_idx]
    O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,args.cluster_idx)
    combinations = find_all_simulation_combinations(O_qubits, rho_qubits, len(cluster_circ.qubits))

    count = int(len(combinations)/num_workers)
    remainder = len(combinations) % num_workers

    if rank == size-1:
        # print('MPI evaluator on cluster %d'%args.cluster_idx)
        cluster_prob = {}
        # bar = pb.ProgressBar(max_value=num_workers)
        for i in range(num_workers):
            state = MPI.Status()
            rank_cluster_prob = comm.recv(source=MPI.ANY_SOURCE,status=state)
            cluster_prob.update(rank_cluster_prob)
            # bar.update(i)
        pickle.dump(cluster_prob, open( '%s/cluster_%d_prob.p'%(dirname,args.cluster_idx), 'wb' ))
    elif rank<remainder:
        combinations_start = rank * (count + 1)
        combinations_stop = combinations_start + count + 1
        rank_combinations = combinations[combinations_start:combinations_stop]
        print('rank %d runs %d combinations for cluster %d'%(rank,len(rank_combinations),args.cluster_idx))
        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
        cluster_circ=cluster_circ,
        combinations=rank_combinations,
        backend=args.backend,noisy=args.noisy)
        comm.send(cluster_prob, dest=size-1)
    else:
        combinations_start = rank * count + remainder
        combinations_stop = combinations_start + (count - 1) + 1
        rank_combinations = combinations[combinations_start:combinations_stop]
        print('rank %d runs %d combinations for cluster %d'%(rank,len(rank_combinations),args.cluster_idx))
        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
        cluster_circ=cluster_circ,
        combinations=rank_combinations,
        backend=args.backend,noisy=args.noisy)
        comm.send(cluster_prob, dest=size-1)