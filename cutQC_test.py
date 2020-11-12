import numpy as np
from qiskit.compiler import transpile
from datetime import datetime

from qiskit_helper_functions.non_ibmq_functions import generate_circ
from qiskit_helper_functions.ibmq_functions import get_device_info

from cutqc.main import CutQC

if __name__ == '__main__':
    small_device_info = get_device_info(token='5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
        hub='ibm-q-ornl',
        group='anl',
        project='csc430',
        device_name='ibmq_bogota',
        fields=['device'],datetime=datetime.now())

    large_device_info = get_device_info(token='5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
        hub='ibm-q-ornl',
        group='anl',
        project='csc430',
        device_name='ibmq_boeblingen',
        fields=['device'],datetime=datetime.now())

    for full_circ_size in [6,8,10]:
        circuit_type = 'aqft'
        max_subcircuit_qubit = 5
        qubit_limit = 10
        circuit_name = '%s_%d'%(circuit_type,full_circ_size)

        circuit = generate_circ(full_circ_size=full_circ_size,circuit_type=circuit_type)
        if circuit.num_qubits==0:
            continue

        cutqc = CutQC(circuits={circuit_name:circuit})
        cutqc.cut(max_subcircuit_qubit=max_subcircuit_qubit, num_subcircuits=[2,3], max_cuts=10)
        # cutqc.evaluate(num_workers=1,eval_mode='sv',early_termination=[1])
        # cutqc.post_process(num_workers=1,eval_mode='sv',early_termination=1,qubit_limit=qubit_limit,recursion_depth=3)
        # cutqc.verify(circuit_name='supremacy_20',max_subcircuit_qubit=max_subcircuit_qubit,early_termination=1,num_workers=1,qubit_limit=qubit_limit,eval_mode='sv')

        if len(cutqc.circuits[circuit_name])==0:
            continue
        mapped_circuit = transpile(cutqc.circuits[circuit_name]['circuit'],backend=large_device_info['device'],layout_method='noise_adaptive')
        print(circuit_name,mapped_circuit.depth())
        subcircuits = cutqc.circuits[circuit_name]['subcircuits']
        for subcircuit_idx, subcircuit in enumerate(subcircuits):
            mapped_circuit = transpile(subcircuit,backend=small_device_info['device'],layout_method='noise_adaptive')
            print(circuit_name,'subcircuit %d'%subcircuit_idx,mapped_circuit.depth())