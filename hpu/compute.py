from hpu.component import ComponentInterface

class COMPUTE(ComponentInterface):
    def __init__(self,config):
        pass

    def run(self,kronecker_terms,flagged_states):
        if len(flagged_states)>0:
            for flagged_state in flagged_states:
                subcircuit_idx = flagged_state['subcircuit_idx']
                subcircuit_instance_idx = flagged_state['subcircuit_instance_idx']
                shot_state = flagged_state['shot_state']
                delta = flagged_state['delta']
                print('%d_%d %d \N{GREEK CAPITAL LETTER DELTA}=%.3f'%(subcircuit_idx,subcircuit_instance_idx,shot_state,delta))
            print()

    def get_output(self, options):
        pass

    def close(self, message: str):
        pass