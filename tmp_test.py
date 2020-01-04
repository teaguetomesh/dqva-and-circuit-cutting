from utils.helper_fun import generate_circ
from utils.mitigation import TensoredMitigation

circ = generate_circ(full_circ_size=8,circuit_type='supremacy')
circ_dict = {'test':{'circ':circ}}

mitigation = TensoredMitigation(circ_dict=circ_dict,device_name='ibmq_boeblingen')
mitigation.run()
mitigation.retrieve()