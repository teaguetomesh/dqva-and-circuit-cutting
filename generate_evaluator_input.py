import pickle
import os
import subprocess
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import evaluator_prob as evaluator
import uniter_prob as uniter
from scipy.stats import wasserstein_distance
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import NoiseAdaptiveLayout
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--fc-shots', metavar='N', type=int,help='number of shots on full circuit')
    parser.add_argument('--cluster-shots', metavar='N', type=int,help='number of shots on clusters')
    args = parser.parse_args()

    num_shots = int(args.fc_shots)
    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    properties = device.properties()
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(properties)
    basis_gates = noise_model.basis_gates

    # times = {'searcher':[],'quantum_evaluator':[],'classical_evaluator':[],'uniter':[]}
    max_qubit = 6
    max_clusters = 6

    dimension_l = [(2,4),[3,3],[3,4]]

    dirname = './noisy_benchmark_data'
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
        # circ = gen_hwea(i*j,1)
        dag = circuit_to_dag(circ)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(dag)
        initial_layout = noise_mapper.property_set['layout']
        
        print('Evaluating sv noiseless fc')
        sv_noiseless_fc = evaluator.simulate_circ(circ=circ,backend='statevector_simulator',noisy=False,qasm_info=None)

        print('Evaluating qasm')
        qasm_info = [None,None,None,num_shots,None]
        qasm_noiseless_fc = evaluator.simulate_circ(circ=circ,backend='qasm_simulator',noisy=False,qasm_info=qasm_info)

        print('Evaluating qasm + noise')
        qasm_info = [noise_model,coupling_map,basis_gates,num_shots,None]
        qasm_noisy_fc = evaluator.simulate_circ(circ=circ,backend='qasm_simulator',noisy=True,qasm_info=qasm_info)

        fc_evaluations = {'sv_noiseless':sv_noiseless_fc,
        'qasm':qasm_noiseless_fc,
        'qasm+noise':qasm_noisy_fc}

        # print('Evaluating qasm + noise + NA')
        # qasm_info = [noise_model,coupling_map,basis_gates,num_shots,initial_layout]
        # qasm_noisy_na_fc = evaluator.simulate_circ(circ=circ,backend='qasm_simulator',noisy=True,qasm_info=qasm_info)

        # fc_evaluations = {'sv_noiseless':sv_noiseless_fc,
        # 'qasm':qasm_noiseless_fc,
        # 'qasm+noise':qasm_noisy_fc,
        # 'qasm+noise+na':qasm_noisy_na_fc}

        # Looking for a cut
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(1,max_clusters+1),hw_max_qubit=max_qubit,evaluator_weight=1)
        searcher_time = time() - searcher_begin
        m.print_stat()

        clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
        print('Complete path map:')
        [print(x,complete_path_map[x]) for x in complete_path_map]

        pickle.dump([dimension,searcher_time,circ,fc_evaluations,clusters,complete_path_map],
        open('{}/evaluator_input_{}_qubits_{}_clusters_{}_{}_shots_{}*{}.p'.format(dirname,max_qubit,max_clusters,num_shots,int(args.cluster_shots),i,j),'wb'))
        print('-'*100)