from cutqc.evaluator import mutate_measurement_basis

from hpu.component import ComponentInterface
from hpu.ppu import PPU
from hpu.nisq import NISQ
from hpu.mux import MUX
from hpu.dram import DRAM

class HPU(ComponentInterface):
    def __init__(self,config):
        self.ppu = PPU(config=config['ppu'])
        self.nisq = NISQ(config=config['nisq'])
        # self.mux = MUX(num_dram=config['num_dram'])
        # self.drams = [DRAM() for dram_unit_index in range(config['num_dram'])]
    
    def run(self,circuit):
        print('--> HPU running <--')
        ppu_output, message = self.ppu.run(circuit=circuit)
        if len(ppu_output)==0:
            self.close(message=message)
        '''
        NOTE: this is emulating an online NISQ device in HPU
        For emulation, we compute all NISQ output then process shot by shot
        In reality, this can be done entirely online
        '''
        self.nisq.run(subcircuits=ppu_output['subcircuits'])
        # for key in nisq_output:
        #     subcircuit_idx,inits,meas = key
        #     mutated_meas = mutate_measurement_basis(meas)
        #     print(nisq_output[key].keys())
            # for subcircuit_output in nisq_output[key]['memory']:
        #         for meas in mutated_meas:
        #             dram_unit_index = self.mux.run(options={'subcircuit_idx':subcircuit_idx,'inits':inits,'meas':meas,'output':subcircuit_output})
        #             self.drams[dram_unit_index].load_input()
        self.close(message='Finished')
    
    def observe(self):
        pass

    def close(self, message):
        print('--> HPU shuts down <--')
        print(message)