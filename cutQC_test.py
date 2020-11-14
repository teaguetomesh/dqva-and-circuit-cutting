from time import time

from qiskit_helper_functions.non_ibmq_functions import generate_circ

from cutqc.main import CutQC

if __name__ == '__main__':
    ibmq = {'token':'5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
            'hub':'ibm-q-ornl',
            'group':'anl',
            'project':'csc430'}
    
    for full_circ_size in [12]:
        circuit_type = 'supremacy'
        max_subcircuit_qubit = 10
        qubit_limit = 10
        circuit_name = '%s_%d'%(circuit_type,full_circ_size)

        circuit = generate_circ(full_circ_size=full_circ_size,circuit_type=circuit_type)
        if circuit.num_qubits==0:
            continue

        cutqc = CutQC(circuits={circuit_name:circuit})
        cutqc.cut(max_subcircuit_qubit=max_subcircuit_qubit, num_subcircuits=[2,3], max_cuts=10)
        evaluate_begin = time()
        cutqc.evaluate(circuit_cases=['%s|%d'%(circuit_name,max_subcircuit_qubit)],
        eval_mode='sv',num_nodes=1,num_threads=2,early_termination=[1],ibmq=ibmq)
        print(time()-evaluate_begin)
        # cutqc.post_process(num_workers=1,eval_mode='sv',early_termination=1,qubit_limit=qubit_limit,recursion_depth=3)
        # cutqc.verify(circuit_name='supremacy_20',max_subcircuit_qubit=max_subcircuit_qubit,early_termination=1,num_workers=1,qubit_limit=qubit_limit,eval_mode='sv')