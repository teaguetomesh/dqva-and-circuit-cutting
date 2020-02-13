import numpy as np
import itertools
from utils.helper_fun import find_cuts_pairs, find_cluster_O_rho_qubit_positions, effective_full_state_corresppndence, smart_cluster_order, find_inits_meas
from time import time

def fake_kron(len_a,len_b,run_kron):
    if run_kron:
        prob_a = 1/len_a
        prob_b = 1/len_b
        arr_a = np.array([prob_a for x in range(len_a)])
        arr_b = np.array([prob_b for x in range(len_b)])
        np.kron(arr_a,arr_b)
        return len_a * len_b, 0
    else:
        assert len_a * len_b <= 2**34
        runtime_exponent = np.log2(len_a * len_b)
        estimated_time = (2e-10)*np.exp(0.8*runtime_exponent)
        return len_a * len_b , estimated_time

def fake_reconstruct(complete_path_map, combinations, full_circ, cluster_circs, cluster_sim_probs, run_kron):
    #[print(x,complete_path_map[x]) for x in complete_path_map]
    O_rho_pairs = find_cuts_pairs(complete_path_map)
    num_cuts = len(O_rho_pairs)
    scaling_factor = np.power(2,num_cuts)
    # print('O rho qubits pairs:',O_rho_pairs)

    reconstructed_prob = np.zeros(2**len(full_circ.qubits))
    if len(combinations)>0:
        # dummy_summation_term = np.array([scaling_factor/len(combinations) for x in range(2**len(full_circ.qubits))])
        dummy_summation_term = scaling_factor/len(combinations)
    else:
        # dummy_summation_term = np.zeros(2**len(full_circ.qubits))
        dummy_summation_term = 0
    correspondence_map = effective_full_state_corresppndence(O_rho_pairs,cluster_circs)
    # print('Effective states, full states correspondence map:')
    # [print('cluster %d' % cluster_idx,correspondence_map[cluster_idx],'\n') for cluster_idx in correspondence_map]
    cluster_O_qubit_positions, cluster_rho_qubit_positions = find_cluster_O_rho_qubit_positions(O_rho_pairs, cluster_circs)
    smart_order = smart_cluster_order(O_rho_pairs, cluster_circs)
    # smart_order = range(len(cluster_circs))

    collapsed_cluster_prob = [{} for c in cluster_circs]
    summation_term_memoization_dict = {}
    total_counter = 0
    collapsed_cluster_prob_memoization_counter = 0
    summation_term_memoization_counter = 0
    kron_calls = 0
    collapse_calls = 0
    total_estimated_kron_time = 0

    for i,s in enumerate(combinations):
        # print('s_{} = {}'.format(i,s))
        clusters_init_meas = find_inits_meas(cluster_circs, O_rho_pairs, s)
        accumulated_clusters_init_meas = ()
        summation_term = None
        for cluster_idx in smart_order:
            total_counter += 1
            # print('Cluster {} inits meas = {}'.format(cluster_idx,clusters_init_meas[cluster_idx]))
            init_meas = tuple(clusters_init_meas[cluster_idx])
            accumulated_clusters_init_meas += init_meas
            if len(accumulated_clusters_init_meas)>2 and accumulated_clusters_init_meas in summation_term_memoization_dict:
                summation_term = summation_term_memoization_dict[accumulated_clusters_init_meas]
                summation_term_memoization_counter += 1
            elif init_meas in collapsed_cluster_prob[cluster_idx]:
                kronecker_term = collapsed_cluster_prob[cluster_idx][init_meas]
                if summation_term != None:
                    summation_term, estimated_time = fake_kron(len_a=summation_term,len_b=kronecker_term,run_kron=run_kron)
                    total_estimated_kron_time += estimated_time
                    kron_calls += 1
                else:
                    summation_term = kronecker_term
                summation_term_memoization_dict[accumulated_clusters_init_meas] = summation_term
                collapsed_cluster_prob_memoization_counter += 1
            else:
                kronecker_term = fake_calculate_cluster(cluster_idx=cluster_idx,
                cluster_probs=cluster_sim_probs[cluster_idx],
                init_meas=clusters_init_meas[cluster_idx],
                O_qubit_positions=cluster_O_qubit_positions[cluster_idx],
                effective_state_tranlsation=correspondence_map[cluster_idx])
                collapse_calls += 1
                if summation_term != None:
                    summation_term, estimated_time = fake_kron(len_a=summation_term,len_b=kronecker_term,run_kron=run_kron)
                    total_estimated_kron_time += estimated_time
                    kron_calls += 1
                else:
                    summation_term = kronecker_term
                collapsed_cluster_prob[cluster_idx][init_meas] = kronecker_term
                summation_term_memoization_dict[accumulated_clusters_init_meas] = summation_term
        reconstructed_prob = reconstructed_prob + dummy_summation_term
        # print('-'*100)
    # print()
    # print('Summation term memoized %d/%d, collapsed_term memoized %d/%d, called kron %d times, collapse %d times'%(
    #     summation_term_memoization_counter,
    # total_counter,collapsed_cluster_prob_memoization_counter,total_counter,kron_calls,collapse_calls))
    if run_kron:
        assert total_estimated_kron_time == 0
    else:
        if len(combinations)>0:
            assert total_estimated_kron_time > 0
    return reconstructed_prob, scaling_factor, smart_order, total_estimated_kron_time

def fake_calculate_cluster(cluster_idx,cluster_probs,init_meas,O_qubit_positions,effective_state_tranlsation):
    multiply_sigma_counter = 0
    # print('O qubit positions:',O_qubit_positions)
    initilizations, measurement = init_meas
    num_effective_states = np.power(2,len(measurement)-len(O_qubit_positions))
    kronecker_term = num_effective_states
    arr_a = np.ones(num_effective_states)
    # print('Cluster %d has %d effective states'%(cluster_idx,num_effective_states))
    meas = tuple([x if x!='Z' else 'I' for x in measurement])
    measurement = tuple(measurement)

    initilizations = [[x] if x == 'zero' else [x+'+',x+'-'] for x in initilizations]
    initilizations = list(itertools.product(*initilizations))
    for init in initilizations:
        # print(init,'initialized to',end=' ')
        sign = 1
        init = list(init)
        for idx,i in enumerate(init):
            if i == 'I+':
                init[idx] = 'zero'
            elif i == 'I-':
                init[idx] = 'one'
            elif i == 'X+':
                init[idx] = 'plus'
            elif i == 'X-':
                init[idx] = 'minus'
                sign *= -1
            elif i == 'Y+':
                init[idx] = 'plus_i'
            elif i == 'Y-':
                init[idx] = 'minus_i'
                sign *= -1
            elif i == 'Z+':
                init[idx] = 'zero'
            elif i == 'Z-':
                init[idx] = 'one'
                sign *= -1
            elif i == 'zero':
                continue
            else:
                raise Exception('Illegal initilization symbol :',i)
        init = tuple(init)
        # print('Cluster %d Evaluate'%cluster_idx,init,measurement)
        
        # sigma_key = (init,meas,tuple([measurement[i] for i in O_qubit_positions]))
        # print('sigma key = ',sigma_key)
        effective_cluster_prob = fake_multiply_sigma(full_cluster_prob_len=cluster_probs[(init,meas)],
        cluster_s=[measurement[i] for i in O_qubit_positions],
        cluster_O_qubit_positions=O_qubit_positions,
        effective_state_tranlsation=effective_state_tranlsation)
        multiply_sigma_counter += 1
        
        # kronecker_term += sign*effective_cluster_prob
        arr_b = np.ones(num_effective_states)
        dummy = arr_a + sign*arr_b
    
    # print('length of effective cluster prob:',len(kronecker_term))
    dummy = np.array(dummy)
    # print('called multiply_sigma %d times'%multiply_sigma_counter)
    return num_effective_states

def fake_multiply_sigma(full_cluster_prob_len,cluster_s,cluster_O_qubit_positions,effective_state_tranlsation):
    # print('full cluster instance prob len = ',len(full_cluster_prob))
    # print('cluster O qubits:',cluster_O_qubit_positions)
    # print('assigned s:',cluster_s)
    if len(cluster_O_qubit_positions) == 0:
        # print('no need to collapse')
        return full_cluster_prob_len
    
    total_num_qubits = int(np.log2(full_cluster_prob_len))
    prob = 1/full_cluster_prob_len
    effective_num_qubits = total_num_qubits - len(cluster_O_qubit_positions)
    if effective_state_tranlsation == None:
        contracted_prob = 0
        for full_state in range(full_cluster_prob_len):
            sigma = 1
            bin_full_state = bin(full_state)[2:].zfill(total_num_qubits)
            for s_i,position in zip(cluster_s,cluster_O_qubit_positions):
                O_measurement = bin_full_state[position]
                if s_i!='I' and O_measurement=='1':
                # if O_measurement=='1':
                    sigma *= -1
            # contributing_term = sigma*full_cluster_prob[full_state]
            contributing_term = sigma*prob
            contracted_prob += contributing_term
        return 1
    else:
        effective_cluster_prob = []
        for effective_state in effective_state_tranlsation:
            # bin_effective_state = bin(effective_state)[2:].zfill(effective_num_qubits)
            effective_state_prob = 0
            full_states = effective_state_tranlsation[effective_state]
            # print('effective state {}, binary {} = '.format(effective_state,bin_effective_state))
            for full_state in full_states:
                bin_full_state = bin(full_state)[2:].zfill(total_num_qubits)
                sigma = 1
                for s_i,position in zip(cluster_s,cluster_O_qubit_positions):
                    O_measurement = bin_full_state[position]
                    # print('s = type {} {}, O measurement = type {} {}'.format(type(s_i),s_i,type(O_measurement),O_measurement))
                    if s_i!='I' and O_measurement=='1':
                        sigma *= -1
                contributing_term = sigma*prob
                effective_state_prob += contributing_term
                # print('full state {}, binary {}, {} * {} = {}'.format(full_state,bin_full_state,full_cluster_prob[full_state],sigma,contributing_term))
                # print('O qubit state {}, full state {}, sigma = {}, index = {}'.format(insertion,full_state,sigma,full_state_index))
                # print(contributing_term)
            # print(' =',effective_state_prob)
            effective_cluster_prob.append(effective_state_prob)
        # print('effective cluster inst prob len = ', len(effective_cluster_prob))
        return 2**effective_num_qubits