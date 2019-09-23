import numpy as np
from scipy.stats import wasserstein_distance
import itertools
import copy
from qcg.generators import gen_supremacy, gen_hwea
import cutter as cutter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from qiskit import BasicAer, execute
from qiskit.quantum_info.states.measures import state_fidelity

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
    else:
        output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return output_prob

def project_sv(cluster_sv,projection):
    projected = []
    for i, sv in enumerate(cluster_sv):
        bin_i = bin(i)[2:].zfill(len(projection))
        pattern_match = True
        for b, p in zip(bin_i, projection):
            b = int(b)
            if b!=p and p!='x':
                pattern_match = False
                break
        if pattern_match:
            projected.append(sv)
    return projected

def find_cluster_cut_qubit(complete_path_map):
    pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for idx, O_qubit in enumerate(path[:-1]):
                rho_qubit = path[idx+1]
                pairs.append((O_qubit,rho_qubit))
    return pairs

def find_projections(complete_path_map,combination):
    pairs = find_cluster_cut_qubit(complete_path_map)
    cluster_qubit_counts = {}
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        for q in path:
            if q[0] not in cluster_qubit_counts:
                cluster_qubit_counts[q[0]] = 1
            else:
                cluster_qubit_counts[q[0]] += 1

def reconstructed_reorder(unordered,complete_path_map):
    print('ordering reconstructed sv')
    ordered  = [0 for sv in unordered]
    cluster_out_qubits = {}
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        if output_qubit[0] in cluster_out_qubits:
            cluster_out_qubits[output_qubit[0]].append((output_qubit[1],input_qubit[1]))
        else:
            cluster_out_qubits[output_qubit[0]] = [(output_qubit[1],input_qubit[1])]
    # print(cluster_out_qubits)
    for cluster_idx in cluster_out_qubits:
        cluster_out_qubits[cluster_idx].sort()
        cluster_out_qubits[cluster_idx] = [x[1] for x in cluster_out_qubits[cluster_idx]]
    print(cluster_out_qubits)
    unordered_qubit_idx = []
    for cluster_idx in sorted(cluster_out_qubits.keys()):
        unordered_qubit_idx += cluster_out_qubits[cluster_idx]
    print(unordered_qubit_idx)
    for idx, sv in enumerate(unordered):
        bin_idx = bin(idx)[2:].zfill(len(unordered_qubit_idx))
        # print('sv bin_idx=',bin_idx)
        ordered_idx = [0 for i in unordered_qubit_idx]
        for jdx, i in enumerate(bin_idx):
            ordered_idx[unordered_qubit_idx[jdx]] = i
        # print(ordered_idx)
        ordered_idx = int("".join(str(x) for x in ordered_idx), 2)
        ordered[ordered_idx] = sv
        # print('unordered %d --> ordered %d'%(idx,ordered_idx),'sv=',sv)
    return ordered

circ = gen_supremacy(2,2,8)
uncut_sv = simulate_circ(circ,'sv')
print('Original uncut circuit:')
print(circ)
uncut_sv = simulate_circ(circ,'sv')
print('Uncut state vector first term:',uncut_sv[0])
print('*'*200)
positions = [(circ.qubits[0],1),(circ.qubits[3],2)]
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
print('Cut at:',positions)
for i,cluster in enumerate(clusters):
    print('cluster %d circuit:'%i)
    print(cluster)
print('Complete path map:')
[print(x,complete_path_map[x]) for x in complete_path_map]
O_rho_pairs = find_cluster_cut_qubit(complete_path_map)
print('Cut qubit pairs:')
[print('O qubit:',x[0],'rho qubit:',x[1]) for x in O_rho_pairs]
print('*'*200)

print('Circuit cutting using state vector simulation:')
combinations = itertools.product(range(2),repeat=len(positions))
sv_reconstruction = [0 for i in range(np.power(2,len(circ.qubits)))]
for combination in combinations:
    print('combination:',combination)
    clusters_copy = copy.deepcopy(clusters)
    for i, c in enumerate(combination):
        O_qubit, rho_qubit = O_rho_pairs[i]
        print('c:',c,'for pair:', O_qubit, rho_qubit)
        if c == 1:
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
    cluster_0_sv = simulate_circ(clusters_copy[0],'sv')
    projection = [combination[0],'x','x']
    cluster_0_sv = project_sv(cluster_0_sv,projection)
    cluster_1_sv = simulate_circ(clusters_copy[1],'sv')
    projection = ['x','x',combination[1]]
    cluster_1_sv = project_sv(cluster_1_sv,projection)
    summation_term = np.kron(cluster_0_sv,cluster_1_sv)
    sv_reconstruction += summation_term
    print('-'*100)
sv_reconstruction = reconstructed_reorder(sv_reconstruction,complete_path_map)
print('reconstruction fidelity=', state_fidelity(uncut_sv,sv_reconstruction))
print('*'*200)

def multiply_sigma(cluster_prob,combination,cluster_idx):
    print('before multiplying sigma:',cluster_prob)
    ret = []
    if cluster_idx == 0:
        if combination[0] > 2:
            ret = [cluster_prob[i] - cluster_prob[i+4] for i in range(int(len(cluster_prob)/2))]
        else:
            ret = [cluster_prob[i] + cluster_prob[i+4] for i in range(int(len(cluster_prob)/2))]
    else:
        if combination[1] > 2:
            ret = [cluster_prob[2*i] - cluster_prob[2*i+1] for i in range(int(len(cluster_prob)/2))]
        else:
            ret = [cluster_prob[2*i] + cluster_prob[2*i+1] for i in range(int(len(cluster_prob)/2))]
    print('after multiplying sigma:',ret,'length = ',len(ret))
    return ret

combinations = itertools.product(range(1,9),repeat=len(positions))
prob_reconstruction = [0 for i in range(np.power(2,len(circ.qubits)))]
uncut_prob = simulate_circ(circ,'prob')
for combination in combinations:
    print('s combination:',combination)
    clusters_copy = copy.deepcopy(clusters)
    for i, s in enumerate(combination):
        O_qubit, rho_qubit = O_rho_pairs[i]
        print('s:',s,'for pair:', O_qubit, rho_qubit)
        if s == 2:
            # Prepare in 1
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
        elif s == 3:
            # Measure in X
            q = clusters_copy[O_qubit[0]].qubits[O_qubit[1]]
            dag = circuit_to_dag(clusters_copy[O_qubit[0]])
            dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            clusters_copy[O_qubit[0]] = dag_to_circuit(dag)
            # Prepare in +
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
        elif s == 4:
            # Measure in X
            q = clusters_copy[O_qubit[0]].qubits[O_qubit[1]]
            dag = circuit_to_dag(clusters_copy[O_qubit[0]])
            dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            clusters_copy[O_qubit[0]] = dag_to_circuit(dag)
            # Prepare in -
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
        elif s == 5:
            # Measure in Y
            q = clusters_copy[O_qubit[0]].qubits[O_qubit[1]]
            dag = circuit_to_dag(clusters_copy[O_qubit[0]])
            dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
            dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            clusters_copy[O_qubit[0]] = dag_to_circuit(dag)
            # Prepare in +i
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
            dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
        elif s == 6:
            # Measure in Y
            q = clusters_copy[O_qubit[0]].qubits[O_qubit[1]]
            dag = circuit_to_dag(clusters_copy[O_qubit[0]])
            dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
            dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            clusters_copy[O_qubit[0]] = dag_to_circuit(dag)
            # Prepare in -i
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
            dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
        elif s == 8:
            # Prepare in 1
            q = clusters_copy[rho_qubit[0]].qubits[rho_qubit[1]]
            dag = circuit_to_dag(clusters_copy[rho_qubit[0]])
            dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            clusters_copy[rho_qubit[0]] = dag_to_circuit(dag)
    # [print(x) for x in clusters_copy]
    cluster_0_prob = simulate_circ(clusters_copy[0],'prob')
    cluster_0_prob = multiply_sigma(cluster_0_prob,combination,0)
    cluster_1_prob = simulate_circ(clusters_copy[1],'prob')
    cluster_1_prob = multiply_sigma(cluster_1_prob,combination,1)
    t_s = np.kron(cluster_0_prob,cluster_1_prob)
    c_s = 1
    for s in combination:
        if s==4 or s==6 or s==8:
            c_s *= -1/2
        else:
            c_s *= 1/2
    summation_term = c_s*t_s
    prob_reconstruction += summation_term
    print('-'*100)
print('First element comparison:', uncut_prob[0], prob_reconstruction[0])
prob_reconstruction = reconstructed_reorder(prob_reconstruction,complete_path_map)
print('probability reconstruction distance:',wasserstein_distance(uncut_prob,prob_reconstruction))