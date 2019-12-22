from qcg.generators import gen_supremacy
from qiskit import Aer, execute
from time import time
from utils.helper_fun import apply_measurement
import multiprocessing
import os

def get_runtime(circuits,num_shots,backend_options):
    print('Backend_options settings :',backend_options)
    begin = time()
    backend = Aer.get_backend('qasm_simulator')
    noiseless_qasm_result = execute(circuits, backend, shots=num_shots, backend_options=backend_options).result()
    x = noiseless_qasm_result.results[0]
    print('parallel_experiments = %d, parallel_shots = %d, parallel_state_update = %d'%(noiseless_qasm_result.metadata['parallel_experiments'],x.metadata['parallel_shots'],x.metadata['parallel_state_update']))
    print(time()-begin)

CPU_cores = multiprocessing.cpu_count()
circ = gen_supremacy(2,2,8)
qc = apply_measurement(circ)
circuits = [qc for i in range(16)]
num_shots = int(1e5)

print('-'*50,'Default','-'*50)
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={})
print('-'*50)

print('-'*50,'Testing parallel threads','-'*50)
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_threads':1})
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_threads':5})
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_threads':CPU_cores})
print('-'*50)

print('-'*50,'Testing parallel experiments','-'*50)
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_experiments':2})
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_experiments':5})
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_experiments':CPU_cores})
print('-'*50)

print('-'*50,'Testing parallel shots','-'*50)
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_shots':1})
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_shots':10})
get_runtime(circuits=circuits,num_shots=num_shots,backend_options={'max_parallel_shots':CPU_cores})
print('-'*50)