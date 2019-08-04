from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import simulator_new as simulator
import numpy as np

circ = gen_supremacy(4,4,8,'71230456')
hardness, positions, K, d, num_cluster, m = searcher.find_cuts(circ,hw_max_qubit=12)
m.print_stat()
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
print('Complete path map:')
[print(x,complete_path_map[x]) for x in complete_path_map]
print('*'*200)

for cluster_idx, cluster in enumerate(clusters):
    simulator.generate_cluster_instances(cluster,cluster_idx,complete_path_map)