import itertools

def find_init_meas(combination, O_rho_pairs, subcircuits):
    # print('Finding init_meas for',combination)
    all_init_meas = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        init = ['zero' for q in range(subcircuit.num_qubits)]
        meas = ['comp' for q in range(subcircuit.num_qubits)]
        all_init_meas[subcircuit_idx] = [init,meas]
    for s, pair in zip(combination, O_rho_pairs):
        O_qubit, rho_qubit = pair
        O_qubit_subcircuit_qubits = subcircuits[O_qubit['subcircuit_idx']].qubits
        rho_qubit_subcircuit_qubits = subcircuits[rho_qubit['subcircuit_idx']].qubits
        all_init_meas[rho_qubit['subcircuit_idx']][0][rho_qubit_subcircuit_qubits.index(rho_qubit['subcircuit_qubit'])] = s
        all_init_meas[O_qubit['subcircuit_idx']][1][O_qubit_subcircuit_qubits.index(O_qubit['subcircuit_qubit'])] = s
    # print(all_init_meas)
    for subcircuit_idx in all_init_meas:
        init = all_init_meas[subcircuit_idx][0]
        init_combinations = []
        for idx, x in enumerate(init):
            if x == 'zero':
                init_combinations.append(['zero'])
            elif x == 'I':
                init_combinations.append(['+zero','+one'])
            elif x == 'X':
                init_combinations.append(['2plus','-zero','-one'])
            elif x == 'Y':
                init_combinations.append(['2plusI','-zero','-one'])
            elif x == 'Z':
                init_combinations.append(['+zero','-one'])
            else:
                raise Exception('Illegal initilization symbol :',x)
        init_combinations = list(itertools.product(*init_combinations))
        meas = all_init_meas[subcircuit_idx][1]
        meas_combinations = []
        for x in meas:
            if x == 'comp':
                meas_combinations.append(['comp'])
            elif x=='I' or x == 'X' or x == 'Y' or x=='Z':
                meas_combinations.append(['+%s'%x])
            else:
                raise Exception('Illegal measurement symbol :',x)
        meas_combinations = list(itertools.product(*meas_combinations))
        subcircuit_init_meas = []
        for init in init_combinations:
            for meas in meas_combinations:
                subcircuit_init_meas.append((tuple(init),tuple(meas)))
        all_init_meas[subcircuit_idx] = subcircuit_init_meas
    # print(all_init_meas)
    return all_init_meas

def get_combinations(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr+1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    basis = ['I','X','Y','Z']
    combinations = list(itertools.product(basis,repeat=len(O_rho_pairs)))
    return O_rho_pairs, combinations

def build(full_circuit, combinations, O_rho_pairs, subcircuits, all_indexed_combinations):
    kronecker_terms = {subcircuit_idx:{} for subcircuit_idx in range(len(subcircuits))}
    summation_terms = []
    for i, combination in enumerate(combinations):
        # print('%d/%d combinations:'%(i+1,len(combinations)),combination)
        summation_term = {}
        all_init_meas = find_init_meas(combination, O_rho_pairs, subcircuits)
        for subcircuit_idx in range(len(subcircuits)):
            subcircuit_kron_term = []
            # print('Subcircuit_%d init_meas ='%subcircuit_idx,all_init_meas[subcircuit_idx])
            for init_meas in all_init_meas[subcircuit_idx]:
                # print('Subcircuit_%d init_meas ='%subcircuit_idx,init_meas)
                coefficient = 1
                init = list(init_meas[0])
                for idx, x in enumerate(init):
                    if x == 'zero':
                        continue
                    elif x == '+zero':
                        init[idx] = 'zero'
                    elif x == '+one':
                        init[idx] = 'one'
                    elif x == '2plus':
                        init[idx] = 'plus'
                        coefficient *= 2
                    elif x == '-zero':
                        init[idx] = 'zero'
                        coefficient *= -1
                    elif x == '-one':
                        init[idx] = 'one'
                        coefficient *= -1
                    elif x =='2plusI':
                        init[idx] = 'plusI'
                        coefficient *= 2
                    else:
                        raise Exception('Illegal initilization symbol :',x)
                meas = list(init_meas[1])
                for idx, x in enumerate(meas):
                    if x == 'comp':
                        continue
                    elif x == '+I':
                        meas[idx] = 'I'
                    elif x == '+Z':
                        meas[idx] = 'Z'
                    elif x =='+X':
                        meas[idx] = 'X'
                    elif x == '+Y':
                        meas[idx] = 'Y'
                    else:
                        raise Exception('Illegal measurement symbol :',x)
                init_meas = (tuple(init),tuple(meas))
                subcircuit_inst_index = all_indexed_combinations[subcircuit_idx][init_meas]
                subcircuit_kron_term.append((coefficient,subcircuit_inst_index))
                # print(coefficient,init_meas)
            subcircuit_kron_term = tuple(subcircuit_kron_term)
            if subcircuit_kron_term not in kronecker_terms[subcircuit_idx]:
                subcircuit_kron_index = len(kronecker_terms[subcircuit_idx])
                kronecker_terms[subcircuit_idx][subcircuit_kron_term] = subcircuit_kron_index
            else:
                subcircuit_kron_index = kronecker_terms[subcircuit_idx][subcircuit_kron_term]
            # print('Subcircuit_%d kron term %d ='%(subcircuit_idx,subcircuit_kron_index),subcircuit_kron_term)
            summation_term[subcircuit_idx] = subcircuit_kron_index
        # print('Summation term =',summation_term,'\n')
        summation_terms.append(summation_term)
    # [print(subcircuit_idx,kronecker_terms[subcircuit_idx]) for subcircuit_idx in kronecker_terms]
    return kronecker_terms, summation_terms