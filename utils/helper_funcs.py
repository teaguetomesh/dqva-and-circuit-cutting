from utils.graph_funcs import is_indset

def strip_ancillas(counts, circ):
    num_anc = len(circ.ancillas)
    new_counts = {}
    for key in counts:
        new_counts[key[num_anc:]] = counts[key]
    return new_counts

def hamming_weight(bitstr):
    return sum([1 for bit in bitstr if bit == '1'])

def gen_binary_str(n, bitstr, ret):
    """
    Generate all binary strings of length n
    """
    if n > 0:
        gen_binary_str(n-1, bitstr + '0', ret)
        gen_binary_str(n-1, bitstr + '1', ret)
    else:
        ret.append(bitstr)
    return ret

def brute_force_search(G, lim=None):
    num_nodes = len((list(G.nodes)))
    bitstrs = gen_binary_str(num_nodes, '', [])
    if lim is not None:
        bitstrs = [b for b in bitstrs if hamming_weight(b) >= lim]
    best_hamming_weight = 0
    for bitstr in bitstrs:
        if is_indset(bitstr, G) and hamming_weight(bitstr) > best_hamming_weight:
            best_hamming_weight = hamming_weight(bitstr)
    best_strs = [b for b in bitstrs if hamming_weight(b) == best_hamming_weight \
                                       and is_indset(b, G)]
    return best_strs, best_hamming_weight


def brute_force_search_memory_efficient(G):
    num_nodes = len((list(G.nodes)))
    best_hamming_weight = 0
    best_strs = []
    counter_lim = int(2**num_nodes)
    counter = 0
    while counter < counter_lim:
        bitstr = f'{counter:0{num_nodes}b}'
        if hamming_weight(bitstr) >= best_hamming_weight and is_indset(bitstr, G):
            best_hamming_weight = hamming_weight(bitstr)
            if len(best_strs) > 0 and best_hamming_weight > hamming_weight(best_strs[0]):
                best_strs = [bitstr]
            else:
                best_strs.append(bitstr)
        counter += 1

    return best_strs, best_hamming_weight


