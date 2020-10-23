from hpu.component import ComponentInterface

class MUX(ComponentInterface):
    def __init__(self):
        pass

    def load_input(self, mux_control):
        print('MUX loads input')
        self.mux_control = mux_control
    
    def run(self,options):
        pass

    def observe(self):
        pass

    def close(self):
        pass