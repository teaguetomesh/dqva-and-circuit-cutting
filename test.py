from hpu.hpu import HPU
from hpu.component import ComponentInterface

from helper_functions.non_ibmq_functions import generate_circ
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    circuit = generate_circ(full_circ_size=5,circuit_type='bv')

    ppu_config = {'max_subcircuit_qubit':3,'num_subcircuits':[2],'max_cuts':10}
    nisq_config = {'token':'5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
    'hub':'ibm-q-ornl',
    'group':'anl',
    'project':'csc430',
    'device_name':'ibmq_bogota',
    'real_device':False}

    hpu_config = {'ppu':ppu_config,'nisq':nisq_config}

    hybrid_processor = HPU(config=hpu_config)
    hybrid_processor.run(circuit=circuit)