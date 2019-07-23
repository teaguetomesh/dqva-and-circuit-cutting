import supremacy_generator as suprem_gen
import cut_searcher as cut_searcher
import cutter
import random
from qiskit.tools.visualization import dag_drawer
from qiskit.converters import circuit_to_dag, dag_to_circuit

def find_ops(fragments, fragment_idx, qubit):
    ops = []
    circ = fragments[fragment_idx]
    for node in circuit_to_dag(circ).topological_op_nodes():
        if qubit in node.qargs:
            translated_qargs = []
            for qarg in node.qargs:
                original_qubit = find_original_qubit(fragment_idx, qarg)
                translated_qargs.append(original_qubit)
            ops.append((node.name, translated_qargs))
    return ops

def find_original_qubit(fragment_idx, qubit):
    for original_qubit in complete_path_map:
        if (fragment_idx, qubit) in complete_path_map[original_qubit]:
            return original_qubit
    return None  

circ = suprem_gen.circuit_generator([5,5,8], random_order = True)

pareto_K_d = cut_searcher.find_pareto_solutions(circ=circ, num_clusters=3)
keys = list(pareto_K_d.keys())
key = random.choice(keys)
pareto_cuts, pareto_grouping = pareto_K_d[key]

fragments, complete_path_map, K, d = cutter.cut_circuit(circ, pareto_cuts)
print('a random pareto solution:')
print(key, pareto_cuts)
# [print(x) for x in pareto_grouping]
print('Complete Path Map:')
[print(x, complete_path_map[x]) for x in complete_path_map]
print('K=%d, d=%d' % (K,d))
print('*'*100)

wrong_cutter = False
for input_qubit in complete_path_map:
    print('checking ops on {}'.format(input_qubit))
    path = complete_path_map[input_qubit]
    fragment_ops = []
    for p in path:
        fragment_idx, fragment_qubit = p
        fragment_ops += find_ops(fragments, fragment_idx, fragment_qubit)
    
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
    print('cutter is correct')