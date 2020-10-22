from hpu.component import ComponentInterface
from hpu.ppu import PPU

class HPU(ComponentInterface):
    def __init__(self,config):
        print('--> New HPU instance <--\nconfigurations : {')
        [print(x,'=',config[x]) for x in config]
        print('}')
        self.ppu = PPU()
    
    def load_input(self,input):
        print('--> HPU loading input <--')
        self.ppu.load_input(circuit=input['circuit'])
    
    def run(self,options):
        print('--> HPU running <--')
        ppu_output = self.ppu.run(options=options['ppu'])
        if len(ppu_output)==0:
            self.close(message='No cuts found')
    
    def observe(self):
        pass

    def close(self, message):
        print('--> HPU shuts down <--')
        print(message)