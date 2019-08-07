from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import simulator
import numpy as np
import pickle

circ = gen_supremacy(4,4,8,'71230456')
hardness, positions, K, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(2,5),hw_max_qubit=10)
m.print_stat()
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
print('Complete path map:')
[print(x,complete_path_map[x]) for x in complete_path_map]
print('*'*200)

pickle.dump( complete_path_map, open( './data/cpm.p', 'wb' ) )
for i, cluster in enumerate(clusters):
    pickle.dump( cluster, open( './data/cluster_%d.p'%i, 'wb' ) )