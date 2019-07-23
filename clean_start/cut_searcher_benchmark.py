import sys
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
import supremacy_generator as suprem_gen
import auto_cut_finder as cut_finder
import cutter
from qiskit.tools.visualization import dag_drawer
import timeit
import matplotlib.pyplot as plt

dimensions = range(2,9)
num_qubits = [np.power(x,2) for x in dimensions]
fixed_depth_times = np.zeros(len(dimensions))

for idx, i in enumerate(dimensions):
    circ = suprem_gen.circuit_generator([i,i,8], random_order = True)
    searcher_start = timeit.default_timer()
    pareto_K_d = cut_finder.find_pareto_solutions(circ=circ, num_clusters=2)
    searcher_end = timeit.default_timer()
    fixed_depth_times[idx] = searcher_end-searcher_start
    for pareto_key in pareto_K_d:
        pareto_K, pareto_d = pareto_key
        pareto_cuts, pareto_grouping = pareto_K_d[pareto_key]
        fragments, complete_path_map, K, d = cutter.cut_circuit(circ, pareto_cuts)
        if K!= pareto_K or d != pareto_d:
            raise Exception('pareto predicted {}, cutter returned {}'.format(pareto_key, (K,d)))

depth = range(8,24,3)
fixed_qubits_times = np.zeros(len(depth))
for idx, d in enumerate(depth):
    circ = suprem_gen.circuit_generator([4,4,d], random_order = True)
    searcher_start = timeit.default_timer()
    pareto_K_d = cut_finder.find_pareto_solutions(circ=circ, num_clusters=2)
    searcher_end = timeit.default_timer()
    fixed_qubits_times[idx] = searcher_end-searcher_start
    for pareto_key in pareto_K_d:
        pareto_K, pareto_d = pareto_key
        pareto_cuts, pareto_grouping = pareto_K_d[pareto_key]
        fragments, complete_path_map, K, d = cutter.cut_circuit(circ, pareto_cuts)
        if K!= pareto_K or d != pareto_d:
            raise Exception('pareto predicted {}, cutter returned {}'.format(pareto_key, (K,d)))
            
            
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.plot(num_qubits, fixed_depth_times)
plt.xlabel('#qubits')
plt.ylabel('running time (s)')
plt.title('Google supremacy crcuits with depth=8')
plt.subplot(122)
plt.plot(depth, fixed_qubits_times)
plt.xlabel('depth')
plt.title('#qubits = 4*4')
plt.savefig('cut_searcher_benchmark.pdf')