from utils.helper_fun import generate_circ, evaluate_circ
from utils.conversions import list_to_dict
import numpy as np
from time import time
from qiskit import Aer, execute

full_circ = generate_circ(full_circ_size=20,circuit_type='supremacy')
backend = Aer.get_backend('statevector_simulator')

begin = time()
job = execute(full_circ, backend=backend)
result = job.result()
outputstate = result.get_statevector(full_circ)
outputstate_dict = list_to_dict(l=outputstate)
for key in outputstate_dict:
    x = outputstate_dict[key]
    outputstate_dict[key] = np.absolute(x)**2
print('%.2f seconds'%(time()-begin))

# ground_truth = evaluate_circ(circ=full_circ,backend='statevector_simulator',evaluator_info=None,force_prob=True)