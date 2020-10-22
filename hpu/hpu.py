from hpu.component import ComponentInterface
from hpu.ppu import PPU
from hpu.nisq import NISQ

class HPU(ComponentInterface):
    def __init__(self,config):
        print('--> New HPU instance <--\nconfigurations : {')
        [print(x,'=',config[x]) for x in config]
        print('}')
        self.ppu = PPU()
        self.nisq = NISQ(token=config['token'],hub=config['hub'],group=config['group'],project=config['project'],device_name=config['device_name'])
    
    def load_input(self,input):
        print('--> HPU loading input <--')
        self.ppu.load_input(circuit=input['circuit'])
    
    def run(self,options):
        print('--> HPU running <--')
        ppu_output, message = self.ppu.run(options=options['ppu'])
        if len(ppu_output)==0:
            self.close(message=message)
        else:
            self.nisq.load_input(subcircuits=ppu_output['subcircuits'])
            self.nisq.run(options=options['nisq'])
        self.close(message='Finished')
    
    def observe(self):
        pass

    def close(self, message):
        print('--> HPU shuts down <--')
        print(message)