from helper_fun import find_saturated_shots
from qcg.generators import gen_supremacy, gen_hwea

circ = gen_supremacy(3,3,8,order='75601234')
saturated_shots = find_saturated_shots(circ)
print(saturated_shots)