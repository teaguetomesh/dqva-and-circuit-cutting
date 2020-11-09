from qiskit_helper_functions.non_ibmq_functions import generate_circ
import numpy as np

from cutqc.main import CutQC

if __name__ == '__main__':
    circuit = generate_circ(full_circ_size=4,circuit_type='supremacy')

    cutqc = CutQC(circuits={'supremacy_4':circuit})
    cutqc.cut(max_subcircuit_qubit=3, num_subcircuits=[2,3], max_cuts=10)
    cutqc.evaluate(num_workers=1,eval_mode='sv',early_termination=[1])
    cutqc.post_process(num_workers=1,eval_mode='sv',early_termination=1,qubit_limit=2,recursion_depth=3)
    cutqc.verify(circuit_name='supremacy_4',max_subcircuit_qubit=3,early_termination=1,num_workers=1,qubit_limit=2,eval_mode='sv')