import sys
import itertools
import time
import numpy as np

sys.path.append('../')

import qsplit.qsplit_mlrecon_methods as qmm


# (1) idetify cut_nodes and uncut_nodes (nodes incident to a cut and their complement)
# (2) choose "hot nodes": nodes incident to a graph partition,
#       to which we will nonetheless apply a partial mixer in the first mixing layer
# WARNING: this algorithm has combinatorial complexity:
#   O({ #cut_nodes_in_subgraph \choose #max_cuts })
# I am simply assuming that this complexity won't be a problem for now
# if it becomes a problem when we scale up, we should rethink this algorithm
def choose_nodes(graph, subgraphs, cut_edges, max_cuts):
    cut_nodes = []
    for edge in cut_edges:
        cut_nodes.extend(edge)

    # collect subgraph data
    subgraph_A, subgraph_B = subgraphs
    cut_nodes_A = [node for node in subgraph_A.nodes if node in cut_nodes]
    cut_nodes_B = [node for node in subgraph_B.nodes if node in cut_nodes]
    subgraph_cut_nodes = [ (subgraph_A, cut_nodes_A), (subgraph_B, cut_nodes_B) ]

    # compute the cost of each choice of hot nodes
    # hot nodes should all be chosen from one subgraph, so loop over subgraph indices
    choice_cost = {}
    for ext_idx in [ 0, 1 ]:
        # ext_graph: subgraph we're "extending" with nodes from the complement graph
        # ext_cut_nodes: cut_nodes in ext_graph
        ext_graph, ext_cut_nodes = subgraph_cut_nodes[ext_idx]

        # adjacent (complement) graph and cut nodes
        adj_graph, adj_cut_nodes = subgraph_cut_nodes[1-ext_idx]

        # determine the number nodes in adj_cut_nodes that we need to "throw out".
        # nodes that are *not* thrown out are attached to ext_graph in the first mixing layer
        num_to_toss = len(adj_cut_nodes) - max_cuts
        num_to_toss = max(num_to_toss,0)

        # determine size of fragments after circuit cutting.
        # if there are several options (of nodes to toss) with the same "cut cost",
        # these fragment sizes are used to choose between those options
        ext_size = ext_graph.number_of_nodes() + len(adj_cut_nodes) - num_to_toss
        complement_size = subgraphs[1-ext_idx].number_of_nodes()
        frag_sizes = tuple(sorted([ ext_size, complement_size ], reverse = True))

        # if we don't need to throw out any nodes,
        # log a choice_cost of 0 and skip the calculation below
        if num_to_toss == 0:
            choice_cost[ext_idx,()] = (0,) + frag_sizes
            continue

        # for some node (in adj_cut_nodes) that we might throw out
        # (i) determine its neighbors in ext_graph
        # (ii) determine the degrees of those neighbors
        # (iii) add up those degrees
        def single_choice_cost(adj_node):
            return sum([ graph.degree[ext_node]
                         for ext_node in graph.neighbors(adj_node)
                         if ext_node in ext_graph ])

        # loop over all combinations of adjacent nodes that we could throw out
        for toss_nodes in itertools.combinations(adj_cut_nodes, num_to_toss):
             _choice_cost = sum([ single_choice_cost(node) for node in toss_nodes ])
             choice_cost[ext_idx,toss_nodes] = (_choice_cost,) + frag_sizes

    # get the index subgraph we're "extending" and the adjacent nodes we're tossing out
    ext_idx, toss_nodes = min(choice_cost, key = choice_cost.get)
    ext_graph, ext_cut_nodes = subgraph_cut_nodes[ext_idx]

    # determine whether a node in ext_graph has any neighbors in toss_nodes
    def _no_tossed_neighbors(ext_node):
        return not any( neighbor in toss_nodes for neighbor in graph.neighbors(ext_node) )

    # hot nodes = those without neighbors that we are tossing out
    hot_nodes = list(filter(_no_tossed_neighbors, ext_cut_nodes))
    return cut_nodes, hot_nodes


def simple_choose_nodes(graph, subgraphs, cut_edges, max_cuts, init_state):
    cut_nodes = []
    for edge in cut_edges:
        cut_nodes.extend(edge)

    # collect subgraph data
    subgraph_A, subgraph_B = subgraphs
    cut_nodes_A = [node for node in subgraph_A.nodes if node in cut_nodes]
    cut_nodes_B = [node for node in subgraph_B.nodes if node in cut_nodes]
    subgraph_cut_nodes = [ (subgraph_A, cut_nodes_A), (subgraph_B, cut_nodes_B) ]

    # Randomly select the subgraph to draw hot nodes from
    rand_index = np.random.choice([0,1])
    cur_subgraph, cur_cut_nodes = subgraph_cut_nodes[rand_index]
    other_subgraph, other_cut_nodes = subgraph_cut_nodes[(rand_index + 1) % 2]

    # Collect all potential hot nodes (i.e. where cost < max_cuts and the node is not already in the MIS)
    valid_hot_nodes = []
    for node in cur_cut_nodes:
        neighbors = list(graph.neighbors(node))
        cost = len([n for n in neighbors if n in other_cut_nodes])
        if cost <= max_cuts and list(reversed(init_state))[node] == '0':
            valid_hot_nodes.append(node)

    np.random.shuffle(valid_hot_nodes)
    hot_nodes = []
    cur_cost = 0
    for node in valid_hot_nodes:
        neighbors = list(graph.neighbors(node))
        temp_cost = len([n for n in neighbors if n in other_cut_nodes])
        if cur_cost + temp_cost <= max_cuts:
            hot_nodes.append(node)
            cur_cost += temp_cost
        if cur_cost == max_cuts:
            break

    return cut_nodes, hot_nodes


def sim_with_cutting(fragments, wire_path_map, frag_shots, backend, mode="likely",
                     verbose=0):
    """
    A helper function to simulate a fragmented circuit.

    Output:
    probs: dict{bitstring : float}
        Outputs a dictionary containing the simulation results. Keys are the
        bitstrings which were observed and their values are the probability that
        they occurred with.
    """

    # build fragment models
    model_time_start = time.time()

    frag_data = qmm.collect_fragment_data(fragments, wire_path_map,
                                          shots = frag_shots,
                                          tomography_backend = backend)
    direct_models = qmm.direct_fragment_model(frag_data)
    if mode == "direct":
        models = direct_models
    elif mode == "likely":
        likely_models = qmm.maximum_likelihood_model(direct_models)
        models = likely_models
    else:
        raise Exception('Unknown recombination mode:', mode)

    model_time = time.time() - model_time_start

    # recombine models to recover full circuit output
    recombine_time_start = time.time()
    recombined_dist = qmm.recombine_fragment_models(models, wire_path_map)
    recombine_time = time.time() - recombine_time_start

    # print timing info
    if verbose:
        print(f"\tModel time: {model_time:.3f}, Recombine time: {recombine_time:.3f}")

    return recombined_dist
