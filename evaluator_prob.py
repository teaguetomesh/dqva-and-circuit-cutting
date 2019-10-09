from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit.circuit.classicalregister import ClassicalRegister
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

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def simulate_circ(circ, simulator, noisy=False, provider_info=None, output_format='prob', num_shots=1024):
    if noisy:
        if output_format!='prob':
            raise Exception('Illegal noisy evaluation method, output_format has to be prob')
        if provider_info==None:
            raise Exception('Provider info is required for noisy evaluation')
        if simulator!='qasm_simulator' and simulator!='ibmq_qasm_simulator':
            raise Exception('Noisy evaluation cannot use {} evaluator'.format(simulator))
    if simulator == 'statevector_simulator':
        backend = Aer.get_backend(simulator)
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_ordered = [0 for sv in outputstate]
        for i, sv in enumerate(outputstate):
            reverse_i = reverseBits(i,len(circ.qubits))
            outputstate_ordered[reverse_i] = sv
        if output_format == 'sv':
            return outputstate_ordered
        elif output_format == 'prob':
            output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
            return output_prob
        else:
            raise Exception('Illegal output format = ',output_format)
    elif simulator == 'qasm_simulator':
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        backend = Aer.get_backend(simulator)
        if not noisy:
            job_sim = execute(qc, backend, shots=num_shots)
            result = job_sim.result()
            counts = result.get_counts(qc)
        else:
            provider, noise_model, coupling_map, basis_gates = provider_info
            result = execute(qc, backend,
                       noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates,shots=num_shots).result()
            counts = result.get_counts(qc)
        prob_ordered = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            prob_ordered[reversed_state] = counts[state]/num_shots
        return prob_ordered
    elif simulator == 'ibmq_qasm_simulator':
        provider, noise_model, coupling_map, basis_gates = provider_info
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        backend = provider.get_backend('ibmq_qasm_simulator')
        result_noise = execute(qc, backend, 
                       noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates,shots=num_shots).result()
        counts_noise = result_noise.get_counts(qc)
        prob_ordered = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in counts_noise:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            prob_ordered[reversed_state] = counts_noise[state]/num_shots
        return prob_ordered
    else:
        raise Exception('Illegal simulator:',simulator)

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

def simulate_clusters(complete_path_map, clusters, provider_info=None, simulator_backend='statevector_simulator',noisy=False):
    all_cluster_prob = []

    for cluster_idx, cluster_circ in enumerate(clusters):
        print('Simulating cluster %d'%cluster_idx)
        print(cluster_circ)
        cluster_prob = {}
        O_qubits, rho_qubits = find_cluster_O_rho_qubits(complete_path_map,cluster_idx)
        combinations = find_all_simulation_combinations(O_qubits, rho_qubits, len(cluster_circ.qubits))
        bar = pb.ProgressBar(max_value=len(combinations))
        for counter, combination in enumerate(combinations):
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
            cluster_inst_prob = simulate_circ(circ=cluster_circ_inst,
            simulator=simulator_backend,
            noisy=noisy,
            provider_info=provider_info,
            output_format='prob',
            num_shots=1024)
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
            bar.update(counter)
        all_cluster_prob.append(cluster_prob)
    return all_cluster_prob

if __name__ == '__main__':
    begin = time()
    dirname = './data'
    complete_path_map = pickle.load(open( '%s/cpm.p'%dirname, 'rb' ))

    [print(x, complete_path_map[x]) for x in complete_path_map]

    cluster_circ_files = [f for f in glob.glob(dirname+'/cluster_*_circ.p')]
    clusters = []
    all_cluster_prob = []
    for cluster_idx in range(len(cluster_circ_files)):
        cluster_circ = pickle.load(open(('%s/cluster_%d_circ.p'%(dirname,cluster_idx)), 'rb'))
        clusters.append(cluster_circ)
    all_cluster_prob = simulate_clusters(complete_path_map, clusters)
    pickle.dump(all_cluster_prob, open('%s/cluster_sim_prob.p'%dirname, 'wb' ))

    full_circ = pickle.load(open(('%s/full_circ.p'%dirname), 'rb'))
    full_circ_sim_prob = simulate_circ(circ=full_circ,
            simulator='statevector_simulator',
            output_format='prob')
    pickle.dump(full_circ_sim_prob, open('%s/full_circ_sim_prob.p'%dirname, 'wb' ))
    print('Python time elapsed = %f seconds'%(time()-begin))