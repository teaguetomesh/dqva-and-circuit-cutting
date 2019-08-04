from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import simulator
import numpy as np

circ = gen_supremacy(3,3,8,'71230456')
hardness, positions, K, d, num_cluster, m = searcher.find_cuts(circ,hw_max_qubit=12)
m.print_stat()
fragments, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
# [print(x,complete_path_map[x]) for x in complete_path_map]
fragment_all_s = simulator.simulate_fragments(fragments, complete_path_map)
print(np.shape(fragment_all_s))