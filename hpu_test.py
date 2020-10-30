from hpu.hpu import HPU

from helper_functions.non_ibmq_functions import generate_circ
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    ppu_config = {'max_subcircuit_qubit':5,'num_subcircuits':[2],'max_cuts':10,'verbose':False}
    nisq_config = {'token':'5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
    'hub':'ibm-q-ornl',
    'group':'anl',
    'project':'csc430',
    'device_name':'ibmq_bogota',
    'real_device':False}
    dram_config = {'dram_directory':'./hpu_test/dram','snapshot_directory':'./hpu_test/snapshot','approximation_threshold':0.1}
    compute_config = {}

    hpu_config = {'ppu':ppu_config,'nisq':nisq_config,'dram':dram_config,'compute':compute_config}

    online_computes = []
    offline_computes = []
    for full_circ_size in [6,8]:
        hybrid_processor = HPU(config=hpu_config)
        circuit = generate_circ(full_circ_size=full_circ_size,circuit_type='bv')
        hybrid_processor.run(circuit=circuit)
        online_compute, offline_compute = hybrid_processor.get_output()
        online_computes.append(online_compute)
        offline_computes.append(offline_compute)
    plt.figure()
    plt.plot([6,8],online_computes,'x-',label='Online CC')
    plt.plot([6,8],offline_computes,'x-',label='Offline CC')
    plt.xlabel('Qubits',size=15)
    plt.xticks(size=15)
    plt.ylabel('Compute Cost',size=15)
    plt.yticks(size=15)
    plt.title('BV')
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig('./test.pdf',dpi=400)
    plt.close()