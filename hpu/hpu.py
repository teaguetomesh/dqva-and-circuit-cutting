import os, subprocess

from hpu.component import ComponentInterface
from hpu.ppu import PPU
from hpu.nisq import NISQ
from hpu.dram import DRAM
from hpu.compute import COMPUTE

class HPU(ComponentInterface):
    def __init__(self,config):
        self.ppu = PPU(config=config['ppu'])
        self.nisq = NISQ(config=config['nisq'])
        self.dram = DRAM(config=config['dram'])
        self.compute = COMPUTE(config=config['compute'])

        dram_directory = config['dram']['dram_directory']
        snapshot_directory = config['dram']['snapshot_directory']
        for directory in [dram_directory,snapshot_directory]:
            if os.path.exists(directory):
                subprocess.run(['rm','-r',directory])
            os.makedirs(directory)
    
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
        shots_ctr = 0
        self.total_compute_required = 0
        while True:
            try:
                shot = next(shot_generator)
            except StopIteration:
                break
            self.dram.run(shot=shot)
            # TODO: frequency of update is hardcoded for now
            if shots_ctr%50==49:
                flagged_states = self.dram.get_output(options={'subcircuit_idx':shot['subcircuit_idx'],'subcircuit_instance_idx':shot['subcircuit_instance_idx'],'top_k':3})
                compute_required = self.compute.run(kronecker_terms=ppu_output['kronecker_terms'],summation_terms=ppu_output['summation_terms'],counter=ppu_output['counter'],flagged_states=flagged_states)
                self.total_compute_required += compute_required
            shots_ctr += 1
        print('Online update requires %d compute, offline requires %d compute'%(self.total_compute_required,ppu_output['cost_estimate']))
        self.offline_cost = ppu_output['cost_estimate']
        self.close(message='Finished')
    
    def get_output(self):
        return self.total_compute_required, self.offline_cost

    def close(self, message):
        print('--> HPU shuts down <--')
        print(message)