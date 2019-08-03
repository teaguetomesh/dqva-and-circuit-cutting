import sys
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qcg.generators import gen_supremacy
import cutter
import MIQCP_searcher
import randomized_searcher as r_s
from qiskit.tools.visualization import dag_drawer
import timeit
import matplotlib.pyplot as plt

dimensions = range(2,7)
num_qubits = [x*x for x in dimensions]
MIQCP_hardness = []
MIQCP_runtime = []

for dimension in dimensions:
    circ = gen_supremacy(dimension,dimension,8)
    start = timeit.default_timer()
    hardness, positions, K, d, num_cluster, model = MIQCP_searcher.find_cuts(circ)
    end = timeit.default_timer()
    MIQCP_runtime.append(end-start)
    if hardness != float('inf'):
        print('{} cuts, {} clusters, hardness = {}, K = {}, d = {}'
        .format(len(positions), num_cluster, hardness, K, d))
        MIQCP_hardness.append(hardness)
    else:
        MIQCP_hardness.append(-1)
        print('none of the clusters is feasible')

num_trials = [int(1e2),int(1e3),int(1e4)]
r_s_hardness = {}
r_s_runtime = {}
for num in num_trials:
    r_s_hardness[num] = []
    r_s_runtime[num] = []
    for dimension in dimensions:
        circ = gen_supremacy(dimension,dimension,8)
        start = timeit.default_timer()
        hardness, positions, K, d, num_cluster = r_s.find_cuts(circ, num_trials=num)
        end = timeit.default_timer()
        r_s_runtime[num].append(end-start)
        if hardness != float('inf'):
            print('{} cuts, {} clusters, hardness = {}, K = {}, d = {}'
            .format(len(positions), num_cluster, hardness, K, d))
            r_s_hardness[num].append(hardness)
        else:
            r_s_hardness[num].append(-1)
            print('randomized searcher found no solution')

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.plot(num_qubits, MIQCP_hardness,label='MIQCP')
for num_trials in r_s_hardness:
    plt.plot(num_qubits, r_s_hardness[num_trials],label='Randomized Searcher %.2e trials'%num_trials)
plt.xlabel('#qubits')
plt.ylabel('clustering hardness (arbitrary units)')
plt.title('Google supremacy crcuits with depth=8')
plt.legend()
plt.subplot(122)
plt.plot(num_qubits, MIQCP_runtime, label='MIQCP')
for num_trials in r_s_runtime:
    plt.plot(num_qubits, r_s_runtime[num_trials], label='Randomized Searcher %.2e trials'%num_trials)
plt.xlabel('#qubits')
plt.ylabel('runtime (s)')
plt.title('Google supremacy crcuits with depth=8')
plt.legend()
plt.savefig('cut_searcher_benchmark.pdf')