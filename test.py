from hpu.hpu import HPU
from hpu.component import ComponentInterface

from helper_functions.non_ibmq_functions import generate_circ
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # circuit = generate_circ(full_circ_size=5,circuit_type='bv')

    # print(circuit)
    
    # hpu_config = {'num_dram':2,
    # 'token':'5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
    # 'hub':'ibm-q-ornl',
    # 'group':'anl',
    # 'project':'csc430',
    # 'device_name':'ibmq_bogota'}
    # ppu_options = {'max_subcircuit_qubit':3,'num_subcircuits':[2],'max_cuts':10}
    # nisq_options = {'real_device':False}

    # hybrid_processor = HPU(config=hpu_config)
    # hybrid_processor.load_input(hpu_input={'circuit':circuit})
    # hybrid_processor.run(options={
    #     'ppu':ppu_options,
    #     'nisq':nisq_options
    #     })

    target_fidelity = 0.01
    nisq_fidelity = np.array([0.1,0.2,0.5])
    num_qubits = np.array(range(10,201,10))
    classical_time = np.exp(num_qubits/10)*target_fidelity
    plt.figure()
    plt.plot(num_qubits,classical_time,'k-',label='classical')
    for x in nisq_fidelity:
        cc_fidelity = target_fidelity/x
        hybrid_time = np.exp(num_qubits/50)*cc_fidelity
        plt.plot(num_qubits,hybrid_time,label='NISQ fidelity = %dX'%(int(x/nisq_fidelity[0])))
    plt.xlabel('Number of qubits')
    plt.ylabel('Runtime')
    plt.yscale('log')
    plt.title('Some Algorithm')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hpu_demonstration.pdf',dpi=400)