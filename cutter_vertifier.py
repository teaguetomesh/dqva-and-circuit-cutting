from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import random
from qiskit.tools.visualization import dag_drawer
from qiskit.converters import circuit_to_dag, dag_to_circuit

def find_ops(cluster_circs, cluster_idx, cluster_qubit_idx, complete_path_map):
    ops = []
    circ = cluster_circs[cluster_idx]
    cluster_qubit = circ.qubits[cluster_qubit_idx]
    for node in circuit_to_dag(circ).topological_op_nodes():
        if cluster_qubit in node.qargs:
            translated_qargs = []
            for qarg in node.qargs:
                cluster_qarg_idx = circ.qubits.index(qarg)
                original_qubit = find_original_qubit(cluster_idx, cluster_qarg_idx, complete_path_map)
                translated_qargs.append(original_qubit)
            ops.append((node.name, translated_qargs))
    return ops

def find_original_qubit(cluster_idx, cluster_qubit_idx, complete_path_map):
    for original_qubit in complete_path_map:
        if (cluster_idx, cluster_qubit_idx) in complete_path_map[original_qubit]:
            return original_qubit
    return None  

circ = gen_supremacy(4,4,8)
hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(2,5),hw_max_qubit=9)

cluster_circs, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
print('Complete Path Map:')
[print(x, complete_path_map[x]) for x in complete_path_map]
print('K={}, d={}'.format(K,d))
print('*'*100)

wrong_cutter = False
for input_qubit in complete_path_map:
    print('checking ops on {}'.format(input_qubit))
    path = complete_path_map[input_qubit]
    fragment_ops = []
    for p in path:
        cluster_idx, cluster_qubit_idx = p
        fragment_ops += find_ops(cluster_circs, cluster_idx, cluster_qubit_idx, complete_path_map)
    
    original_ops = []
    for node in circuit_to_dag(circ).topological_op_nodes():
        if input_qubit in node.qargs:
            original_ops.append((node.name, node.qargs))
    if fragment_ops != original_ops:
        wrong_cutter = True
        print('input_qubit', input_qubit)
        print(fragment_ops)
        print(original_ops)
        print('*'*100)

if not wrong_cutter:
    print('cutter is CORRECT')
else:
    print('cutter is WRONG')