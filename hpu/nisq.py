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

    def run(self,subcircuits):
        self.nisq_scheduler = Scheduler(circ_dict=subcircuits,token=self.token,hub=self.hub,group=self.group,project=self.project,device_name=self.device_name)
        self.nisq_scheduler.run(real_device=self.real_device)
        self.nisq_scheduler.retrieve(save_memory=True,force_prob=True)
    
    def get_output(self, all_indexed_combinations):
        for key in self.nisq_scheduler.circ_dict:
            subcircuit_idx, inits, meas = key
            inits_meas = (tuple(inits),tuple(meas))
            subcircuit_instance_index = all_indexed_combinations[subcircuit_idx][inits_meas]
            memory = self.nisq_scheduler.circ_dict[key]['memory']
            for shot_bitstring in memory:
                yield {'subcircuit_idx':subcircuit_idx, 'subcircuit_instance_index':subcircuit_instance_index, 'shot_bitstring':shot_bitstring}

    def close(self, message):
        pass