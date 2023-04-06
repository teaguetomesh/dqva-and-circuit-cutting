from typing import List, Tuple
import itertools
import networkx as nx
import numpy as np
import qcopt
import qiskit
from scipy.optimize import minimize


def optimize_circuit(circuit):

    backend = qiskit.Aer.get_backend("aer_simulator_statevector")
    def f(params: List):
        # Bind the parameters
        bound_circ = circuit.bind_parameters({theta: param for theta, param in zip(circuit.parameters, params)})

        # Compute the cost function
        bound_circ.save_statevector()

        result = qiskit.execute(bound_circ, backend=backend).result()
        probs = qiskit.quantum_info.Statevector(result.get_statevector(bound_circ)).probabilities_dict(decimals=7)

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        # print('Expectation value:', avg_cost)
        return -avg_cost

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=len(circuit.parameters))
    out = minimize(f, x0=init_params, method="COBYLA")
    return out


def get_circuit_from_graph(G, max_cuts, params, barriers=0, decompose_level=2):
    # Partition the graph
    partition = nx.algorithms.community.kernighan_lin_bisection(G)
    subgraphs, cut_edges = get_subgraphs(G, partition)

    # identify the subgraph of every node
    subgraph_dict = {}
    for i, subgraph in enumerate(subgraphs):
        for qubit in subgraph:
            subgraph_dict[qubit] = i

    # Select hot nodes
    cut_nodes, hot_nodes = simple_choose_nodes(G, partition, cut_edges, max_cuts)

    # Put active mixers at the front of the order
    cur_permutation = sort_mixers(G, subgraph_dict)
    active_mixers = [node for node in cur_permutation if node not in cut_nodes] + hot_nodes
    inactive_mixers = [node for node in cur_permutation if node not in active_mixers]
    active_mixer_order = [node for node in cur_permutation if node in active_mixers] + inactive_mixers

    if not params:
        #params = [(i+1) * np.pi/4 for i in range(len(active_mixers))] + [np.pi/2]
        params = [qiskit.circuit.Parameter(f'theta_{i}') for i in range(len(active_mixers) + 1)]

    circuit = qcopt.qlsa.gen_qlsa(
        G,
        P=1,
        params=params,
        barriers=barriers,
        init_state='0'*len(G.nodes),
        decompose_toffoli=decompose_level,
        mixer_order=active_mixer_order,
        param_lim=len(active_mixers)+1,
    )
    return circuit, subgraph_dict, cut_nodes, hot_nodes, active_mixers

def get_subgraphs(G, partition):
    subgraphs = []
    cut_edges = []
    all_edges = G.edges
    for subgraph_nodes in partition:
        subG = nx.Graph()
        subG.add_nodes_from(subgraph_nodes)

        for v1, v2 in all_edges:
            if v1 in subgraph_nodes and v2 in subgraph_nodes:
                subG.add_edge(v1, v2)
            if v1 in subgraph_nodes and v2 not in subgraph_nodes:
                cut_edges.append((v1, v2))

        subgraphs.append(subG)

    return subgraphs, cut_edges

def _cut_cost(graph, partition, hot_nodes, subgraph_dict, cut_nodes):
    subgraph_appearances = {cut_node: [] for cut_node in cut_nodes}

    for subgraph_idx, subgraph_nodes in enumerate(partition):
        for subgraph_node in subgraph_nodes:
            if (subgraph_node in cut_nodes) and (subgraph_node not in hot_nodes):
                continue
            for partial_mixer_node in list(graph.neighbors(subgraph_node)) + [subgraph_node]:
                if partial_mixer_node in cut_nodes:
                    subgraph_appearances[partial_mixer_node].append(subgraph_idx)

    num_cuts = 0
    for cut_node, appearances in subgraph_appearances.items():
        cut_node_subgraph = subgraph_dict[cut_node]
        for subgraph in set(appearances):
            if cut_node_subgraph != subgraph:
                num_cuts += 1
    return num_cuts

def _is_connected(graph, partition, hot_nodes, subgraph_dict):
    meta_graph = nx.Graph()
    for i in range(len(partition)):
        meta_graph.add_node(i)

    for hot_node in hot_nodes:
        for edge in graph.edges:
            if hot_node in edge:
                subgraph_i = subgraph_dict[edge[0]]
                subgraph_j = subgraph_dict[edge[1]]
                if subgraph_i != subgraph_j:
                    meta_graph.add_edge(subgraph_i, subgraph_j)

    return nx.is_connected(meta_graph)

def simple_choose_nodes(graph: nx.Graph, partition: List[List[int]],
                        cut_edges: List[Tuple[int, int]], max_cuts: int,
                        ) -> Tuple[List[int], List[int]]:
    subgraph_dict = {}
    for i, subgraph_nodes in enumerate(partition):
        for node in subgraph_nodes:
            subgraph_dict[node] = i

    cut_nodes = []
    for edge in cut_edges:
        cut_nodes.extend(edge)
    cut_nodes = list(set(cut_nodes))

    # Generate the set of all possible hot node sets
    # NOTE: this is extremely inefficient, for n hot nodes,
    # there are n-choose-1 + n-choose-2 + ... n-choose-n possible hot node sets
    all_possible_hot_nodes = []
    for r in range(1, len(cut_nodes)+1):
        if r > max_cuts:
            break
        for length_r_hot_nodes in itertools.combinations(cut_nodes, r):
            all_possible_hot_nodes.append(length_r_hot_nodes)

    # Eliminate those hot node sets that result in a disconnected graph
    all_connected_hot_nodes = []
    for possible_hot_nodes in all_possible_hot_nodes:
        if _is_connected(graph, partition, possible_hot_nodes, subgraph_dict):
            all_connected_hot_nodes.append(possible_hot_nodes)

    # Eliminate those hot node sets that require cuts > max_cuts
    all_feasible_hot_nodes = []
    for possible_hot_nodes in all_connected_hot_nodes:
        cost = _cut_cost(graph, partition, possible_hot_nodes, subgraph_dict, cut_nodes)
        if cost <= max_cuts:
            all_feasible_hot_nodes.append(possible_hot_nodes)

    # For now, uniform random sampling
    hot_nodes = all_feasible_hot_nodes[np.random.choice([i for i in range(len(all_feasible_hot_nodes))])]

    return cut_nodes, list(hot_nodes)

def sort_mixers(G, subgraph_dict):
    cur_mixer_order = list(G.nodes)

    # Group the nodes by subgraph
    subgraph_nodes = [[] for _ in range(len(set(subgraph_dict.values())))]
    for node in cur_mixer_order:
        subgraph_nodes[subgraph_dict[node]].append(node)

    # Order mixers sequentially in each subgraph
    new_mixer_order = []
    for sublist in subgraph_nodes:
        new_mixer_order.extend(sorted(sublist))

    return new_mixer_order


