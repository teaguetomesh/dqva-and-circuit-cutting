from hpu.component import ComponentInterface
from hpu.ppu import PPU
from hpu.nisq import NISQ
from hpu.dram import DRAM

class HPU(ComponentInterface):
    def __init__(self,config):
        self.ppu = PPU(config=config['ppu'])
        self.nisq = NISQ(config=config['nisq'])
        self.dram = DRAM(config=config['dram'])
    
    def run(self,circuit):
        print('--> HPU running <--')
        self.ppu.run(circuit=circuit)
        ppu_output = self.ppu.get_output()
        if len(ppu_output)==0:
            self.close(message='PPU found no cut solutions')
        '''
        NOTE: this is emulating an online NISQ device in HPU
        For emulation, we compute all NISQ output then process shot by shot
        In reality, this can be done entirely online
        '''
        self.nisq.run(subcircuits=ppu_output['subcircuit_instances'])
        shot_generator = self.nisq.get_output(all_indexed_combinations=ppu_output['all_indexed_combinations'])
        while True:
            try:
                shot = next(shot_generator)
            except StopIteration:
                break
            self.dram.run(shot=shot)
        self.close(message='Finished')
    
    def get_output(self):
        pass

    def close(self, message):
        print('--> HPU shuts down <--')
        print(message)