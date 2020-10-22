from hpu.hpu import HPU
from hpu.component import ComponentInterface

from helper_functions.non_ibmq_functions import generate_circ

if __name__ == '__main__':
    circuit = generate_circ(full_circ_size=5,circuit_type='bv')
    
    hybrid_processor = HPU(config={'num_mememory':4})
    hybrid_processor.load_input(input={'circuit':circuit})
    hybrid_processor.run(options={
        'ppu':{'max_subcircuit_qubit':3,'num_subcircuits':[2],'max_cuts':0}
        })