from utils.helper_fun import generate_circ, get_evaluator_info, evaluate_circ, apply_measurement
import utils.MIQCP_searcher as searcher
import utils.cutter as cutter
from time import time

for fc_size in range(60,81,10):
    circ = generate_circ(full_circ_size=fc_size,circuit_type='supremacy')
    max_clusters = 3
    cluster_max_qubit = fc_size - 10
    searcher_begin = time()
    hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=circ,reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
    num_clusters=range(2,min(len(circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)
    searcher_time = time() - searcher_begin

    if m != None:
        # m.print_stat()
        clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
        print('%d-on-%d'%(fc_size,cluster_max_qubit))
        print('{:d} cuts --> {}, searcher time = {}'.format(K,d,searcher_time))
    else:
        print('NOT feasible')