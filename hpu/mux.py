from hpu.component import ComponentInterface

class MUX(ComponentInterface):
    def __init__(self, num_memory):
        self.num_memory = num_memory

    def load_input(self, mux_control):
        print('MUX loads input')
        self.mux_control = mux_control
    
    def run(self,options):
        subcircuit_instance_ctr = self.mux_control[options['subcircuit_idx']][(options['inits'],options['meas'])]

    def observe(self):
        pass

    def close(self):
        pass