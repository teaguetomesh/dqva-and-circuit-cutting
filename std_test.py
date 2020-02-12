from utils.helper_fun import generate_circ, evaluate_circ
from time import time

full_circ = generate_circ(full_circ_size=20,circuit_type='supremacy')

begin = time()
ground_truth = evaluate_circ(circ=full_circ,backend='statevector_simulator',evaluator_info=None,force_prob=True)
print('%.2f seconds'%(time()-begin))
