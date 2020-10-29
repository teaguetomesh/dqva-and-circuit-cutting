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
        snapshot_pckl = '%s/%d_%d.pckl'%(self.snapshot_directory,subcircuit_idx,subcircuit_instance_index)
        snapshot_subcircuit_instance_output = read_dict(filename=snapshot_pckl)
        snapshot_total_shots = snapshot_subcircuit_instance_output['total_shots'] if 'total_shots' in snapshot_subcircuit_instance_output else 0
        
        dram_pckl = '%s/%d_%d.pckl'%(self.dram_directory,subcircuit_idx,subcircuit_instance_index)
        subcircuit_instance_output = read_dict(filename=dram_pckl)
        dram_total_shots = subcircuit_instance_output['total_shots']

        for shot_state in subcircuit_instance_output:
            if shot_state=='total_shots':
                continue
            snapshot_num_shots = snapshot_subcircuit_instance_output[shot_state] if shot_state in snapshot_subcircuit_instance_output else 0
            snapshot_p = snapshot_num_shots/snapshot_total_shots if snapshot_total_shots>0 else 0

            dram_num_shots = subcircuit_instance_output[shot_state]
            dram_p = dram_num_shots/dram_total_shots

            delta = abs(dram_p - snapshot_p)/snapshot_p if snapshot_p>0 else 1
            if delta>self.approximation_threshold:
                snapshot_subcircuit_instance_output['total_shots'] = snapshot_total_shots + dram_num_shots - snapshot_num_shots
                snapshot_subcircuit_instance_output[shot_state] = dram_num_shots
                print('subcircuit instance %d_%d state %d : %d(%.3f) --> %d(%.3f), total shots : %d --> %d'%(
                    subcircuit_idx,subcircuit_instance_index,shot_state,snapshot_num_shots,snapshot_p,dram_num_shots,dram_p,snapshot_total_shots,snapshot_subcircuit_instance_output['total_shots']))
        pickle.dump(snapshot_subcircuit_instance_output, open(snapshot_pckl,'wb'))

    def close(self):
        pass