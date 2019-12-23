from qcg.generators import gen_supremacy
from qiskit import Aer, execute
from time import time
from utils.helper_fun import apply_measurement, cross_entropy, reverseBits, evaluate_circ
import multiprocessing
import os
import numpy as np

def get_runtime(circuits,num_shots,backend_options):
    print('Backend_options settings :',backend_options)
    eval_begin = time()
    backend = Aer.get_backend('qasm_simulator')
    noiseless_qasm_result = execute(circuits, backend, shots=num_shots, backend_options=backend_options).result()
    x = noiseless_qasm_result.results[0]
    print('parallel_experiments = %d, parallel_shots = %d, parallel_state_update = %d'%(noiseless_qasm_result.metadata['parallel_experiments'],x.metadata['parallel_shots'],x.metadata['parallel_state_update']))
    print('Eval time = %.3f'%(time()-eval_begin))

    noiseless_probs = []
    for circ_idx, circuit in enumerate(circuits):
        noiseless_counts = noiseless_qasm_result.get_counts(circ_idx)
        assert sum(noiseless_counts.values())>=num_shots
        noiseless_prob = [0 for x in range(np.power(2,len(circuit.qubits)))]
        reverse_begin = time()
        for state in noiseless_counts:
            reversed_state = reverseBits(int(state,2),len(circuit.qubits))
            noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
        print('Reverse time = %.3f'%(time()-reverse_begin))
        noiseless_probs.append(noiseless_prob)

        sv_begin = time()
        ground_truth = evaluate_circ(circ=circuit,backend='statevector_simulator',evaluator_info=None)
        print('SV time = %.3f'%(time()-sv_begin))
        ce_begin = time()
        ce = cross_entropy(target=ground_truth,obs=noiseless_prob)
        print('CE time = %.3f'%(time()-ce_begin))
    return noiseless_probs

CPU_cores = multiprocessing.cpu_count()
circ = gen_supremacy(4,5,8)
qc = apply_measurement(circ)
circuits = [qc for i in range(1)]
num_shots = int(1e5)

print('-'*50,'Default','-'*50)
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={})
print('-'*50)

# print('-'*50,'Testing parallel threads','-'*50)
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_threads':1})
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_threads':5})
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_threads':CPU_cores})
# print('-'*50)

# print('-'*50,'Testing parallel experiments','-'*50)
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_experiments':2})
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_experiments':5})
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_experiments':CPU_cores})
# print('-'*50)

# print('-'*50,'Testing parallel shots','-'*50)
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_shots':1})
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_shots':10})
# get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_shots':CPU_cores})
# print('-'*50)