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
        subcircuit_instance_idx = shot['subcircuit_instance_idx']

        dram_pckl = '%s/%d_%d.pckl'%(self.dram_directory,subcircuit_idx,subcircuit_instance_idx)
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
        # TODO: need more efficient probability change detection algorithm
        subcircuit_idx = options['subcircuit_idx']
        subcircuit_instance_idx = options['subcircuit_instance_idx']
        top_k = options['top_k']
        
        dram_pckl = '%s/%d_%d.pckl'%(self.dram_directory,subcircuit_idx,subcircuit_instance_idx)
        dram_output = read_dict(filename=dram_pckl)
        dram_total_shots = dram_output['total_shots']

        snapshot_pckl = '%s/%d_%d.pckl'%(self.snapshot_directory,subcircuit_idx,subcircuit_instance_idx)
        snapshot_output = read_dict(filename=snapshot_pckl)

        flagged_states = []
        for shot_state in dram_output:
            if shot_state=='total_shots':
                continue
            else:
                dram_p = dram_output[shot_state]/dram_total_shots
                snapshot_p = snapshot_output[shot_state] if shot_state in snapshot_output else 0
                # TODO: what criteria of delta to use?
                # delta = abs(dram_p-snapshot_p)/snapshot_p if snapshot_p>0 else float('inf')
                delta = dram_p-snapshot_p
                if abs(delta)>self.approximation_threshold:
                    # print('%d: %.3f --> %.3f'%(shot_state,snapshot_p,dram_p))
                    flagged_states.append({'subcircuit_idx':subcircuit_idx,
                    'subcircuit_instance_idx':subcircuit_instance_idx,
                    'shot_state':shot_state,'delta':delta})
                    snapshot_output[shot_state] = dram_p
        if subcircuit_idx>0:
            flagged_states = sorted(flagged_states,key=lambda x:abs(x['delta']))
            flagged_states = flagged_states[:top_k] if len(flagged_states)>top_k else flagged_states
        else:
            flagged_states = []
        pickle.dump(snapshot_output, open(snapshot_pckl,'wb'))
        return flagged_states

    def close(self):
        pass