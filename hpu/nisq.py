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
        self.nisq_device = Scheduler(circ_dict=subcircuits,token=self.token,hub=self.hub,group=self.group,project=self.project,device_name=self.device_name)
        self.nisq_device.run(real_device=self.real_device)
        self.nisq_device.retrieve(save_memory=True,force_prob=False)
        return self.nisq_device.circ_dict
    
    def observe(self):
        pass

    def close(self, message):
        pass