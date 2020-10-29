import os
import subprocess
import pickle

from helper_functions.non_ibmq_functions import read_dict

from hpu.component import ComponentInterface

class DRAM(ComponentInterface):
    def __init__(self, config):
        self.dram_directory = config['dram_directory']
        if not os.path.exists(self.dram_directory):
            os.makedirs(self.dram_directory)
        else:
            subprocess.run(['rm','-r',self.dram_directory])
            os.makedirs(self.dram_directory)
        self.snapshot_directory = config['snapshot_directory']
        self.approximation_threshold = config['approximation_threshold']

    def run(self, shot):
        subcircuit_idx = shot['subcircuit_idx']
        subcircuit_instance_index = shot['subcircuit_instance_index']
        shot_birstring = shot['shot_bitstring']
        shot_state = int(shot_birstring,2)

        dram_pckl = '%s/%d_%d.pckl'%(self.dram_directory,subcircuit_idx,subcircuit_instance_index)
        subcircuit_instance_output = read_dict(filename=dram_pckl)
        if int(shot_birstring,2) in subcircuit_instance_output:
            subcircuit_instance_output[int(shot_birstring,2)] += 1
        else:
            subcircuit_instance_output[int(shot_birstring,2)] = 1
        if 'total_shots' in subcircuit_instance_output:
            subcircuit_instance_output['total_shots'] += 1
        else:
            subcircuit_instance_output['total_shots'] = 1
        curr_total = subcircuit_instance_output['total_shots']
        curr_shot = subcircuit_instance_output[int(shot_birstring,2)]
        curr_p = curr_shot/curr_total
        pickle.dump(subcircuit_instance_output, open(dram_pckl,'wb'))

        snapshot_pckl = '%s/%d_%d.pckl'%(self.snapshot_directory,subcircuit_idx,subcircuit_instance_index)
        snapshot_subcircuit_instance = read_dict(filename=snapshot_pckl)
        snapshot_total = snapshot_subcircuit_instance['total_shots'] if 'total_shots' in snapshot_subcircuit_instance else 0
        snapshot_shot = snapshot_subcircuit_instance[int(shot_birstring,2)] if int(shot_birstring,2) in snapshot_subcircuit_instance else 0
        snapshot_p = snapshot_shot/snapshot_total if snapshot_total!= 0 else 0
        percent_change_toward_1 = (curr_p-snapshot_p)/(1-snapshot_p)
        if percent_change_toward_1 > self.approximation_threshold:
            print('%d/%d --> %d/%d, %.3f closer to 1'%(snapshot_shot,snapshot_total,curr_shot,curr_total,percent_change_toward_1))

    def get_output(self, options):
        pass

    def close(self):
        pass