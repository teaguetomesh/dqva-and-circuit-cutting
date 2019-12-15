from qcg.generators import gen_supremacy
from qiskit import Aer, execute
from time import time
from utils.helper_fun import apply_measurement
import multiprocessing
import os

def get_runtime(circuits,num_shots,max_parallel_threads=0,max_parallel_experiments=1,max_parallel_shots=1):
    print('max_parallel_experiments = %d, max_parallel_shots = %d, max_parallel_threads = %d'%(max_parallel_experiments,max_parallel_shots,max_parallel_threads))
    begin = time()
    backend = Aer.get_backend('qasm_simulator')
    backend_options = {'method': 'automatic','max_parallel_threads':max_parallel_threads,
    'max_parallel_experiments':max_parallel_experiments,'max_parallel_shots':max_parallel_shots}
    noiseless_qasm_result = execute(circuits, backend, shots=num_shots, backend_options=backend_options).result()
    print('parallel_experiments = %d'%noiseless_qasm_result.metadata['parallel_experiments'])
    for x in noiseless_qasm_result.results:
        print('parallel_shots = %d, parallel_state_update = %d'%(x.metadata['parallel_shots'],x.metadata['parallel_state_update']))
    print(time()-begin)
    print('-'*50)

CPU_cores = multiprocessing.cpu_count()
circ = gen_supremacy(2,2,8)
qc = apply_measurement(circ)
circuits = [qc for i in range(1)]
num_shots = int(1e6)

get_runtime(circuits=circuits,num_shots=num_shots,max_parallel_threads=0,max_parallel_experiments=1,max_parallel_shots=1)
get_runtime(circuits=circuits,num_shots=num_shots,max_parallel_threads=0,max_parallel_experiments=3,max_parallel_shots=1)
get_runtime(circuits=circuits,num_shots=num_shots,max_parallel_threads=0,max_parallel_experiments=CPU_cores,max_parallel_shots=1)
get_runtime(circuits=circuits,num_shots=num_shots,max_parallel_threads=0,max_parallel_experiments=1,max_parallel_shots=CPU_cores)

# backend = Aer.get_backend('qasm_simulator')
# backend_options = {'method': 'automatic','max_parallel_experiments':os.cpu_count()}
# begin = time()
# noiseless_qasm_result = execute(circuits, backend, shots=num_shots, backend_options=backend_options).result()
# print(time()-begin)