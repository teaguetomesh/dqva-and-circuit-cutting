from hpu.component import ComponentInterface
from hpu.ppu import PPU
from hpu.nisq import NISQ
from hpu.mux import MUX

class HPU(ComponentInterface):
    def __init__(self,config):
        print('--> New HPU instance <--\nconfigurations : {')
        [print(x,'=',config[x]) for x in config]
        print('}')
        self.ppu = PPU()
        self.nisq = NISQ(token=config['token'],hub=config['hub'],group=config['group'],project=config['project'],device_name=config['device_name'])
        self.mux = MUX()
    
    def load_input(self,hpu_input):
        print('--> HPU loading input <--')
        self.ppu.load_input(circuit=hpu_input['circuit'])
    
    def run(self,options):
        print('--> HPU running <--')
        ppu_output, message = self.ppu.run(options=options['ppu'])
        if len(ppu_output)==0:
            self.close(message=message)
        self.nisq.load_input(subcircuits=ppu_output['subcircuits'])
        nisq_output = self.nisq.run(options=options['nisq'])
        self.mux.load_input(mux_control=ppu_output['mux_control'])
        for key in nisq_output:
            subcircuit_idx,inits,meas = key
            for subcircuit_output in nisq_output[key]['memory']:
                self.mux.run(options={'subcircuit_idx':subcircuit_idx,'inits':inits,'meas':meas,'output':subcircuit_output})
        self.close(message='Finished')
    
    def observe(self):
        pass

    def close(self, message):
        print('--> HPU shuts down <--')
        print(message)