from utils.helper_fun import generate_circ
from utils.mitigation import LocalMitigation, TensoredMitigation

circ = generate_circ(full_circ_size=3,circuit_type='supremacy')
circ_dict = {'test':{'circ':circ}}

# local_mitigation = LocalMitigation(circ_dict=circ_dict,device_name='ibmq_boeblingen')
# local_mitigation.run()
# local_mitigation.retrieve()
# circ_dict = local_mitigation.circ_dict
# print(circ_dict['test']['calibration_matrix'])

tensored_mitigation = TensoredMitigation(circ_dict=circ_dict,device_name='ibmq_boeblingen')
tensored_mitigation.run()
tensored_mitigation.retrieve()