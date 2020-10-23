from hpu.component import ComponentInterface

class MUX(ComponentInterface):
    def __init__(self, num_dram):
        self.num_dram = num_dram

    def load_input(self, mux_control):
        print('MUX loads input')
        self.mux_control = mux_control
    
    def run(self,options):
        subcircuit_instance_ctr = self.mux_control[options['subcircuit_idx']][(options['inits'],options['meas'])]
        dram_unit_index = subcircuit_instance_ctr%self.num_dram
        return dram_unit_index

    def observe(self):
        pass

    def close(self):
        pass