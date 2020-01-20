import numpy as np

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def list_to_dict(l):
    l_dict = {}
    num_qubits = int(np.log(len(l))/np.log(2))
    assert 2**num_qubits == len(l)
    for state, entry in enumerate(l):
        bin_state = bin(state)[2:].zfill(num_qubits)
        l_dict[bin_state] = entry
    assert sum(l_dict.values()) == sum(l)
    return l_dict

def dict_to_array(distribution_dict,force_prob):
    state = list(distribution_dict.keys())[0]
    num_qubits = len(state)
    num_shots = sum(distribution_dict.values())
    cnts = np.zeros(2**num_qubits,dtype=float)
    for state in distribution_dict:
        cnts[int(state,2)] = distribution_dict[state]
    print(sum(cnts),num_shots)
    assert sum(cnts) == num_shots
    if not force_prob:
        return cnts
    else:
        prob = cnts / num_shots
        assert abs(sum(prob)-1)<1e-10
        return prob

def memory_to_dict(memory):
    mem_dict = {}
    for m in memory:
        if m in mem_dict:
            mem_dict[m] += 1
        else:
            mem_dict[m] = 1
    assert sum(mem_dict.values()) == len(memory)
    return mem_dict