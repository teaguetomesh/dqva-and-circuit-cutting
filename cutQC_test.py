from time import time

from qiskit_helper_functions.non_ibmq_functions import generate_circ

from cutqc.main import CutQC

if __name__ == '__main__':
    ibmq = {'token':'5a928096df5e3c865028e0fc0908fb7c324846d5f135c0d1db304639fa2f701d919fc0cdbcd5824104e28cbc695d7a7993fd38887c1b286af56acd6a21653e78',
            'hub':'ibm-q-ornl',
            'group':'anl',
            'project':'csc430'}
    
    circuits = {}
    circuit_cases = []
    for full_circ_size in [12,15,16]:
        circuit_type = 'supremacy'
        max_subcircuit_qubit = 10
        qubit_limit = 10
        circuit_name = '%s_%d'%(circuit_type,full_circ_size)

        circuit = generate_circ(full_circ_size=full_circ_size,circuit_type=circuit_type)
        if circuit.num_qubits==0:
            continue
        else:
            circuits[circuit_name] = circuit
            circuit_cases.append('%s|%d'%(circuit_name,max_subcircuit_qubit))

    cutqc = CutQC(circuits=circuits,max_subcircuit_qubit=max_subcircuit_qubit, num_subcircuits=[2,3], max_cuts=10)
    cutqc.evaluate(circuit_cases=circuit_cases,eval_mode='runtime',num_nodes=1,num_threads=1,early_termination=[1],ibmq=ibmq)
    # cutqc.post_process(circuit_cases=['%s|%d'%(circuit_name,max_subcircuit_qubit)],
    #     eval_mode='sv',num_nodes=1,num_threads=2,early_termination=1,qubit_limit=qubit_limit,recursion_depth=3)
    # cutqc.verify(circuit_cases=['%s|%d'%(circuit_name,max_subcircuit_qubit)],
    # early_termination=1,num_threads=2,qubit_limit=qubit_limit,eval_mode='sv')