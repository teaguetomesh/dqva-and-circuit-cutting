import pickle
import os
import subprocess
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
from helper_fun import simulate_circ, cross_entropy, find_saturated_shots
import uniter_prob as uniter
from scipy.stats import wasserstein_distance
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import NoiseAdaptiveLayout
import argparse
import datetime as dt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--max-qubit', metavar='N', type=int,help='QC number of qubits limit')
    parser.add_argument('--max-clusters', metavar='N', type=int,help='max number of clusters to split into')
    args = parser.parse_args()

    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    properties = device.properties(dt.datetime(day=16, month=10, year=2019, hour=20))
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties)
    basis_gates = noise_model.basis_gates

    max_qubit = args.max_qubit
    max_clusters = args.max_clusters

    # NOTE: toggle circuits to benchmark
    dimension_l = [[3,3],[2,5]]

    dirname = './benchmark_data'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    for dimension in dimension_l:
        i,j = dimension
        if i*j<=max_qubit:
            continue
        print('-'*100)
        print('%d * %d supremacy circuit'%(i,j))

        # Generate a circuit
        circ = gen_supremacy(i,j,8,order='75601234')
        dag = circuit_to_dag(circ)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(dag)
        initial_layout = noise_mapper.property_set['layout']
        
        print('Evaluating sv noiseless fc')
        sv_noiseless_fc = simulate_circ(circ=circ,backend='statevector_simulator',qasm_info=None)
        identical_dist_ce = cross_entropy(target=sv_noiseless_fc,obs=sv_noiseless_fc)

        print('Evaluating qasm')
        # Deciding how many shots is needed, minimum is 1000
        num_shots = find_saturated_shots(circ)
        qasm_info = [None,None,None,None,None,num_shots]
        qasm_noiseless_fc = simulate_circ(circ=circ,backend='noiseless_qasm_simulator',qasm_info=qasm_info)
        print('requires %.3e shots'%num_shots)

        print('Evaluating qasm + noise')
        qasm_info = [device, properties,coupling_map,noise_model,basis_gates,num_shots]
        qasm_noisy_fc = simulate_circ(circ=circ,backend='noisy_qasm_simulator',qasm_info=qasm_info)

        fc_evaluations = {'sv_noiseless':sv_noiseless_fc,
        'qasm':qasm_noiseless_fc,
        'qasm+noise':qasm_noisy_fc}

        # Looking for a cut
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(1,max_clusters+1),hw_max_qubit=max_qubit,evaluator_weight=1)
        searcher_time = time() - searcher_begin
        m.print_stat()

        clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
        print('Complete path map:')
        [print(x,complete_path_map[x]) for x in complete_path_map]

        pickle.dump([dimension,num_shots,searcher_time,circ,fc_evaluations,clusters,complete_path_map],
        open('{}/evaluator_input_{}_qubits_{}_clusters_{}*{}.p'.format(dirname,args.max_qubit,args.max_clusters,i,j),'wb'))
        print('-'*100)