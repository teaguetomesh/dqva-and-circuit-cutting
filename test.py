from hpu.hpu import HPU
from hpu.component import ComponentInterface

from helper_functions.non_ibmq_functions import generate_circ

if __name__ == '__main__':
    circuit = generate_circ(full_circ_size=5,circuit_type='bv')
    
    hpu_config = {'num_mememory':4,
    'token':'5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
    'hub':'ibm-q-ornl',
    'group':'anl',
    'project':'csc430',
    'device_name':'ibmq_bogota'}
    ppu_options = {'max_subcircuit_qubit':3,'num_subcircuits':[2],'max_cuts':10}
    nisq_options = {'real_device':False}

    hybrid_processor = HPU(config=hpu_config)
    hybrid_processor.load_input(input={'circuit':circuit})
    hybrid_processor.run(options={
        'ppu':ppu_options,
        'nisq':nisq_options
        })