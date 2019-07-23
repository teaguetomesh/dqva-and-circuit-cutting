from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import random
import math
import copy
from collections import Counter
from datetime import datetime
import sys
import numpy as np

class Graph(object):
    def __init__(self, vlist):
        self.verts = {v[0]:Counter(v[1:]) for v in vlist}
        self.update_edges()

    def update_edges(self):
        self.edges = []
        
        for k, v in self.verts.items():
            self.edges += ([(k, t) for t in v.keys() for n in range(v[t]) if k < t])

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        head_key, tail_key = self.edges[edge_index]
        # print('removing edge ', self.edges[edge_index], head_key, tail_key)
        
        head = self.verts[head_key]
        tail = self.verts[tail_key]

        # print('merging head =', head_key, head, 'tail =', tail_key, tail)

        # Remove the edge between head and tail
        del head[tail_key]
        del tail[head_key]

        # print('After removing head->tail, head =', head, 'tail =', tail)

        # Merge tails
        head.update(tail)

        # print('After merging tail, head =', head_key, head)

        # Update all the neighboring vertices of the fused vertex
        for i in tail.keys():
            v = self.verts[i]
            # print('tail neighboring vertex', i, 'vertex', v)
            v[head_key] += v[tail_key]
            del v[tail_key]
            # print('updated to', v)
            
        # Finally remove the tail vertex
        del self.verts[tail_key]

        self.update_edges()
        # print('FINISHED edges:', self.edges)
        # print('FINISHED verts:', self.verts)

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def circuit_to_graph(stripped_circ):
    input_qubit_itr = {}
    for x in stripped_circ.qubits:
        input_qubit_itr[x] = 0
    stripped_dag = circuit_to_dag(stripped_circ)
    vert_info = []
    for vertex in stripped_dag.topological_op_nodes():
        vertex_name = ''
        for qarg in vertex.qargs:
            if vertex_name == '':
                vertex_name += '%s[%d]%d' % (qarg[0].name,qarg[1],input_qubit_itr[qarg])
                input_qubit_itr[qarg] += 1
            else:
                vertex_name += ',%s[%d]%d' % (qarg[0].name,qarg[1],input_qubit_itr[qarg])
                input_qubit_itr[qarg] += 1
        vert_info.append((vertex_name, vertex.qargs))
    
    abstraction = []
    for idx, (vertex_name, vertex_qargs) in enumerate(vert_info):
        abstraction_item = [vertex_name]
        for qarg in vertex_qargs:
            prev_vertex_name, prev_found = find_neighbor(qarg, vert_info[:idx][::-1])
            next_vertex_name, next_found = find_neighbor(qarg, vert_info[idx+1:])
            if prev_found:
                abstraction_item.append(prev_vertex_name)
            if next_found:
                abstraction_item.append(next_vertex_name)
        abstraction.append(abstraction_item)
    graph = Graph(abstraction) 
    return graph

def translate_edge_index(original_r, original_g, curr_g, grouping):
    # print('translating edge index', original_r)
    original_head, original_tail = original_g.edges[original_r]
    # print('original head =', original_head, 'original tail =', original_tail)
    # print('current graph:', curr_g.verts)
    curr_head = None
    if original_head in curr_g.verts:
        curr_head = original_head
    else:
        for group in grouping:
            if original_head in group.split(' '):
                curr_head = group.split(' ')[0]
    curr_tail = None
    if original_tail in curr_g.verts:
        curr_tail = original_tail
    else:
        for group in grouping:
            if original_tail in group.split(' '):
                curr_tail = group.split(' ')[0]
    curr_r = -1
    for edge_idx, edge in enumerate(curr_g.edges):
        if (curr_head, curr_tail) == edge:
            curr_r = edge_idx
    return curr_r

def contract(graph, min_v=2):
    g = copy.deepcopy(graph)
    grouping = [x for x in g.verts]
    contracted_edges = []
    contraction_order = random.sample(range(0,graph.edge_count), graph.edge_count)
    i = 0
    # print('initial grouping:', grouping)
    while g.vertex_count > min_v:
        # r = random.randrange(0, g.edge_count)
        # head, tail = g.edges[r]
        # graph_r = random.randrange(0, graph.edge_count)
        graph_r = contraction_order[i]
        i+=1
        contracted_edges.append(graph.edges[graph_r])
        g_r = translate_edge_index(original_r=graph_r, original_g=graph, curr_g=g, grouping=grouping)
        g_head, g_tail = g.edges[g_r]
        # print('contracting', graph.edges[graph_r], 'in the original graph')
        # print('keep the edge', (g_head, g_tail), 'in the contracted graph')
        g.merge_vertices(g_r)
        
        hi = -1
        ti = -1
        for group_idx, group in enumerate(grouping):
            if g_head in group:
                hi = group_idx
            if g_tail in group:
                ti = group_idx
        grouping.append(grouping[hi] + ' ' + grouping[ti])
        del grouping[min(hi,ti)]
        del grouping[max(hi,ti)-1]
        # print('updated grouping:', grouping)
        # print('*'*100)

    cut_edges = []
    for edge in graph.edges:
        head, tail = edge
        contracted = False
        for group in grouping:
            if head in group.split(' ') and tail in group.split(' '):
                contracted = True
                break
        if not contracted:
            cut_edges.append(edge)

    return g, grouping, cut_edges

def find_crevices(l):
    rolling_idx = l[0]
    d = 1
    for idx, ele in enumerate(l):
        if ele == rolling_idx:
            rolling_idx += 1
        else:
            d += 1
            rolling_idx = ele + 1
    return d

def cluster_character(graph, grouping, hw_max_qubit=24):
    K = graph.edge_count
    max_d = 0
    cumulative_hardness = 0
    for group in grouping:
        group_qubits = {}
        for vertex in group.split(' '):
            qargs = vertex.split(',')
            for qarg in qargs:
                qubit = qarg.split(']')[0] + ']'
                multi_Qgate_idx = int(qarg.split(']')[1])
                if qubit not in group_qubits:
                    group_qubits[qubit] = [multi_Qgate_idx]
                else:
                    group_qubits[qubit].append(multi_Qgate_idx)
        # print(group_qubits)
        group_d = 0
        group_K = 0
        for qubit in group_qubits:
            l = sorted(group_qubits[qubit])
            group_d += find_crevices(l)
            group_K += find_crevices(l) - 1
        # print('K = %d, d = %d' % (K, d))
        group_hardness = float('inf') if group_d > hw_max_qubit else np.power(2,group_d)*np.power(8,group_K)
        cumulative_hardness += math.log(group_hardness)
        max_d = max(max_d, group_d)
    return K, max_d, cumulative_hardness

def find_neighbor(qarg, vert_info):
    for idx, (vertex_name, qargs) in enumerate(vert_info):
        if qarg in qargs:
            return vertex_name, True
    return None, False

def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        source, dest = position
        source_qargs = [x[:len(x)-1] for x in source.split(',')]
        dest_qargs = [x[:len(x)-1] for x in dest.split(',')]
        qubit_cut = list(set(source_qargs).intersection(set(dest_qargs)))
        if len(qubit_cut)>1:
            raise Exception('one cut is cutting on multiple qubits')
        for x in source.split(','):
            if x[:len(x)-1] == qubit_cut[0]:
                source_idx = int(x[len(x)-1])
        for x in dest.split(','):
            if x[:len(x)-1] == qubit_cut[0]:
                dest_idx = int(x[len(x)-1])
        multi_Q_gate_idx = max(source_idx, dest_idx)
        # print('cut qubit:', qubit_cut[0], 'after %d multi qubit gate'% multi_Q_gate_idx)
        wire = None
        for qubit in circ.qubits:
            if qubit[0].name == qubit_cut[0].split('[')[0] and qubit[1] == int(qubit_cut[0].split('[')[1].split(']')[0]):
                wire = qubit
        tmp = 0
        all_Q_gate_idx = None
        for gate_idx, gate in enumerate(list(dag.nodes_on_wire(wire=wire, only_ops=True))):
            if len(gate.qargs)>1:
                tmp += 1
                if tmp == multi_Q_gate_idx:
                    all_Q_gate_idx = gate_idx
        positions.append((wire, all_Q_gate_idx))
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions

# Karger's Algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nC2 logn times
def min_cut(graph, min_v=2, hw_max_qubit=20):
    m = graph.edge_count
    n = graph.vertex_count
    min_hardness = float('inf')
    min_hardness_cuts = None
    min_hardness_K = None
    min_hardness_d = None
    print('splitting into %d fragments, %d edges %d vertices graph will run %d times' % (min_v, m, n, int(n * (n-1) * math.log(n)/2)))
    # TODO: figure out how many trials actually required
    # for i in range(int(n * (n-1) * math.log(n)/2)):
    for i in range(1000):
        random.seed(datetime.now())
        g, grouping, cut_edges = contract(graph, min_v)
        K, d, hardness = cluster_character(g, grouping, hw_max_qubit)
        if hardness < min_hardness:
            min_hardness = hardness
            min_hardness_cuts = cut_edges
            min_hardness_K = K
            min_hardness_d = d
    return min_hardness_cuts, min_hardness, min_hardness_K, min_hardness_d

def _fast_min_cut(graph):
    if graph.vertex_count <= 6:
        return min_cut(graph)
    else:
        t = math.floor(1 + graph.vertex_count / math.sqrt(2))
        
        g1, grouping1, cut_edges1 = contract(graph, t)

        g2, grouping2, cut_edges2 = contract(graph, t)

        min_hardness1, min_hardness_cuts1 = _fast_min_cut(g1)
        min_hardness2, min_hardness_cuts2 = _fast_min_cut(g2)

        if min_hardness1 < min_hardness2:
            return min_hardness1, cut_edges1 + min_hardness_cuts1
        else:
            return min_hardness2, cut_edges2 + min_hardness_cuts2

def find_best_cuts(circ, hw_max_qubit=20,num_clusters=[2]):
    stripped_circ = circ_stripping(circ)
    graph = circuit_to_graph(stripped_circ)
    min_hardness = float('inf')
    best_cuts = None
    best_K = None
    best_d = None
    for i in num_clusters:
        cuts, hardness, K, d = min_cut(graph, i, hw_max_qubit)
        if cuts != None:
            if hardness < min_hardness:
                best_cuts = cuts
                min_hardness = hardness
                best_K = K
                best_d = d
                best_num_clusters = i

    if best_cuts == None:
        raise Exception('Did not find cuts for hw_max_qubit = %d' %hw_max_qubit)
    best_cuts = cuts_parser(best_cuts, circ)
    return best_cuts, min_hardness, best_K, best_d, best_num_clusters