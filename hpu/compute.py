from hpu.component import ComponentInterface

class COMPUTE(ComponentInterface):
    def __init__(self,config):
        pass

    def run(self,kronecker_terms,summation_terms,counter,flagged_states):
        compute_required = 0
        if len(flagged_states)>0:
            affected_subcircuit_instances = {}
            for flagged_state in flagged_states:
                subcircuit_idx = flagged_state['subcircuit_idx']
                subcircuit_instance_idx = flagged_state['subcircuit_instance_idx']
                shot_state = flagged_state['shot_state']
                delta = flagged_state['delta']
                print('subcircuit %d instance %d state %d \N{GREEK CAPITAL LETTER DELTA}=%.3f'%(subcircuit_idx,subcircuit_instance_idx,shot_state,delta))
                if (subcircuit_idx,subcircuit_instance_idx) in affected_subcircuit_instances:
                    affected_subcircuit_instances[(subcircuit_idx,subcircuit_instance_idx)] += 1
                else:
                    affected_subcircuit_instances[(subcircuit_idx,subcircuit_instance_idx)] = 1
            for key in affected_subcircuit_instances:
                subcircuit_idx,subcircuit_instance_idx = key
                num_states_changed = affected_subcircuit_instances[key]
                affected_subcircuit_kron_terms = []
                for subcircuit_kron_term in kronecker_terms[subcircuit_idx]:
                    for item in subcircuit_kron_term:
                        if subcircuit_instance_idx==item[1]:
                            affected_subcircuit_kron_terms.append(kronecker_terms[subcircuit_idx][subcircuit_kron_term])
                for affected_subcircuit_kron_term_idx in affected_subcircuit_kron_terms:
                    # print('subcircuit %d kron_term %d is affected'%(subcircuit_idx,affected_subcircuit_kron_term_idx))
                    for summation_term_idx, summation_term in enumerate(summation_terms):
                        if summation_term[subcircuit_idx] == affected_subcircuit_kron_term_idx:
                            # print('summation term %d is affected'%(summation_term_idx))
                            # FIXME: makeshift solution only works for 2 subcircuits
                            compute_required += num_states_changed*(2**counter[0]['effective'])
            # [print('subcircuit_%d'%subcircuit_idx,kronecker_terms[subcircuit_idx]) for subcircuit_idx in kronecker_terms]
            # [print(summation_term) for summation_term in summation_terms]
            # print()
        return compute_required

    def get_output(self, options):
        pass

    def close(self, message: str):
        pass