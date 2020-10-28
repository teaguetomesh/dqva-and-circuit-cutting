from helper_functions.schedule import Scheduler

from hpu.component import ComponentInterface

class NISQ(ComponentInterface):
    def __init__(self, config):
        self.token = config['token']
        self.hub = config['hub']
        self.group = config['group']
        self.project = config['project']
        self.device_name = config['device_name']
        self.real_device = config['real_device']

    def process(self,subcircuits):
        self.nisq_device = Scheduler(circ_dict=subcircuits,token=self.token,hub=self.hub,group=self.group,project=self.project,device_name=self.device_name)
        self.nisq_device.run(real_device=self.real_device)
        self.nisq_device.retrieve(save_memory=True,force_prob=True)
    
    def run(self, all_indexed_combinations):
        for key in self.nisq_device.circ_dict:
            subcircuit_idx, inits, meas = key
            inits_meas = (tuple(inits),tuple(meas))
            combination_index = all_indexed_combinations[subcircuit_idx][inits_meas]
            memory = self.nisq_device.circ_dict[key]['memory']
            for shot_bitstring in memory:
                yield {'subcircuit_idx':subcircuit_idx, 'combination_index':combination_index, 'shot_bitstring':shot_bitstring}
    
    def observe(self):
        pass

    def close(self, message):
        pass