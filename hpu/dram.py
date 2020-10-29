import pickle

from helper_functions.non_ibmq_functions import read_dict

from hpu.component import ComponentInterface

class DRAM(ComponentInterface):
    def __init__(self, config):
        self.dram_directory = config['dram_directory']
        self.snapshot_directory = config['snapshot_directory']
        self.approximation_threshold = config['approximation_threshold']

    def run(self, shot):
        subcircuit_idx = shot['subcircuit_idx']
        subcircuit_instance_index = shot['subcircuit_instance_index']
        dram_pckl = '%s/%d_%d.pckl'%(self.dram_directory,subcircuit_idx,subcircuit_instance_index)
        subcircuit_instance_output = read_dict(filename=dram_pckl)
        if 'total_shots' in subcircuit_instance_output:
            subcircuit_instance_output['total_shots'] += 1
        else:
            subcircuit_instance_output['total_shots'] = 1

        shot_birstring = shot['shot_bitstring']
        shot_state = int(shot_birstring,2)
        if shot_state in subcircuit_instance_output:
            subcircuit_instance_output[shot_state] += 1
        else:
            subcircuit_instance_output[shot_state] = 1
        pickle.dump(subcircuit_instance_output, open(dram_pckl,'wb'))

    def get_output(self, options):
        subcircuit_idx = options['subcircuit_idx']
        subcircuit_instance_index = options['subcircuit_instance_index']
        
        dram_pckl = '%s/%d_%d.pckl'%(self.dram_directory,subcircuit_idx,subcircuit_instance_index)
        subcircuit_instance_output = read_dict(filename=dram_pckl)
        dram_total_shots = subcircuit_instance_output['total_shots']

        for shot_state in subcircuit_instance_output:
            if shot_state=='total_shots':
                continue
            else:
                print('%d: %d/%d'%(shot_state,subcircuit_instance_output[shot_state],dram_total_shots))

    def close(self):
        pass