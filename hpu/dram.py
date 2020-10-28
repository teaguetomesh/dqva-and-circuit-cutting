import os
import subprocess

from hpu.component import ComponentInterface

class DRAM(ComponentInterface):
    def __init__(self, config):
        self.save_directory = config['save_directory']
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        else:
            subprocess.run(['rm','-r',self.save_directory])
            os.makedirs(self.save_directory)

    def run(self, shot):
        print(shot)

    def observe(self):
        pass

    def close(self):
        pass