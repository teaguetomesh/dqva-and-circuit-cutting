from helper_functions.schedule import Scheduler

from hpu.component import ComponentInterface

class NISQ(ComponentInterface):
    def __init__(self, token, hub, group, project, device_name):
        self.token = token
        self.hub = hub
        self.group = group
        self.project = project
        self.device_name = device_name

    def load_input(self,subcircuits):
        print('NISQ loads input')
        self.nisq_device = Scheduler(circ_dict=subcircuits,token=self.token,hub=self.hub,group=self.group,project=self.project,device_name=self.device_name)
    
    def run(self,options):
        self.nisq_device.run(real_device=options['real_device'])
        self.nisq_device.retrieve(force_prob=False)
        return self.nisq_device.circ_dict
    
    def observe(self):
        pass

    def close(self, message):
        pass