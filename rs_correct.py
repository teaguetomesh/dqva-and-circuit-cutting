from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
import random
import math
import copy
from collections import Counter
from datetime import datetime
import sys
import numpy as np
from qcg.generators import gen_supremacy

class Graph(object):
    def __init__(self, vlist):
        self.verts = {v[0]:Counter(v[1:]) for v in vlist}
        self.update_edges()

    def update_edges(self):
        self.edges = []
        
        for k, v in self.verts.items():
            self.edges += ([(k, t) for t in v.keys() for n in range(v[t])])

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        head_key, tail_key = self.edges[edge_index]
        # print('removing edge ', head_key, tail_key)
        
        head = self.verts[head_key]
        tail = self.verts[tail_key]

        # print('merging head =', head_key, head, 'tail =', tail_key, tail)

        # Remove the edge from head to tail
        del head[tail_key]

        # print('After removing head->tail, head =', head, 'tail =', tail)

        # Merge tail into head
        head.update(tail)

        # print('After merging tail, head =', head_key, head)

        # Vertices pointing to tail should now point to head
        for vertex in self.verts:
            # print(vertex, self.verts[vertex])
            if tail_key in self.verts[vertex]:
                if head_key in self.verts[vertex]:
                    self.verts[vertex][head_key] += self.verts[vertex][tail_key]
                else:
                    self.verts[vertex][head_key] = self.verts[vertex][tail_key]
                del self.verts[vertex][tail_key]
            
        # Finally remove the tail vertex
        # print(self.verts)
        # print('remove ', tail_key, self.verts[tail_key], 'from verts')
        del self.verts[tail_key]
        # print(self.verts)

        self.update_edges()
        # print('FINISHED merging, edges:', self.edges)
        # print('FINISHED merging, verts:', self.verts)

def circuit_to_graph(circ):
    input_qubit_itr = {}
    for x in circ.qubits:
        input_qubit_itr[x] = 0
    stripped_dag = circuit_to_dag(circ)
    vert_info = []
    for vertex in stripped_dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception('vertex does not have exactly 2 arguments')
        vertex_name = '%s[%d]%d %s[%d]%d' % (vertex.qargs[0][0].name,vertex.qargs[0][1],input_qubit_itr[vertex.qargs[0]],
        vertex.qargs[1][0].name,vertex.qargs[1][1],input_qubit_itr[vertex.qargs[1]])
        input_qubit_itr[vertex.qargs[0]] += 1
        input_qubit_itr[vertex.qargs[1]] += 1
        vert_info.append((vertex_name, vertex.qargs))

    abstraction = []
    for idx, (vertex_name, vertex_qargs) in enumerate(vert_info):
        abstraction_item = [vertex_name]
        for qarg in vertex_qargs:
            # prev_vertex_name, prev_found = find_neighbor(qarg, vert_info[:idx][::-1])
            next_vertex_name, next_found = find_neighbor(qarg, vert_info[idx+1:])
            # if prev_found:
            #     abstraction_item.append(prev_vertex_name)
            if next_found:
                abstraction_item.append(next_vertex_name)
        abstraction.append(abstraction_item)
    graph = Graph(abstraction) 
    return graph

def find_neighbor(qarg, vert_info):
    for idx, (vertex_name, qargs) in enumerate(vert_info):
        if qarg in qargs:
            return vertex_name, True
    return None, False

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def min_cut(graph, min_v=2, hw_max_qubit=20):
    m = graph.edge_count
    n = graph.vertex_count
    min_hardness = float('inf')
    min_hardness_cuts = None
    min_hardness_K = None
    min_hardness_d = None
    # print('splitting into %d fragments, %d edges %d vertices graph will run %d times' % (min_v, m, n, int(n * (n-1) * math.log(n)/2)))
    print('splitting into %d fragments, %d edges %d vertices graph' % (min_v, m, n))
    # TODO: figure out how many trials actually required
    # for i in range(int(n * (n-1) * math.log(n)/2)):
    for i in range(1):
        random.seed(datetime.now())
        g, grouping, cut_edges = contract(graph, min_v)
        print('characterizing for trial %d'%i)
        K, d, hardness = cluster_character(g, grouping, cut_edges, hw_max_qubit)
        if hardness < min_hardness:
            min_hardness = hardness
            min_hardness_cuts = cut_edges
            min_hardness_K = K
            min_hardness_d = d
    return min_hardness_cuts, min_hardness, min_hardness_K, min_hardness_d

def contract(graph, min_v=2):
    g = copy.deepcopy(graph)
    grouping = [x for x in g.verts]
    contraction_order = random.sample(range(0,graph.edge_count), graph.edge_count)
    i = 0
    # print('Initial grouping:', grouping)
    # print('Initial graph verts:', graph.verts)
    print('Initial graph edges:', graph.edges)
    # print('*'*100)
    while g.vertex_count > min_v:
        graph_r = contraction_order[i]
        g_r = translate_edge_index(original_edge=graph.edges[graph_r], curr_g=g, grouping=grouping)

        print('keep edge', graph.edges[graph_r], 'in the original graph')
        print('contracting edge', g.edges[g_r], 'in the contracted graph')

        g.merge_vertices(g_r)
        
        g_head, g_tail = g.edges[g_r]
        hi = -1
        ti = -1
        for group_idx, group in enumerate(grouping):
            if g_head in group.split(';'):
                hi = group_idx
            if g_tail in group.split(';'):
                ti = group_idx
        if hi!=ti:
            grouping.append(grouping[hi] + ';' + grouping[ti])
            del grouping[min(hi,ti)]
            del grouping[max(hi,ti)-1]
        print('updated grouping:')
        [print(x) for x in grouping]
        # print('updated graph edges:', g.edges)
        # print('updated graph verts:', g.verts)
        # print('*'*100)
        i+=1

    cut_edges = []
    for edge in graph.edges:
        head, tail = edge
        contracted = False
        for group in grouping:
            if head in group.split(';') and tail in group.split(';'):
                contracted = True
                break
        if not contracted:
            cut_edges.append(edge)
    print('*'*50,'finished contraction','*'*50)
    print('cut_edges are:', cut_edges)
    return g, grouping, cut_edges

def translate_edge_index(original_edge, curr_g, grouping):
    original_head, original_tail = original_edge
    
    return curr_r

def cluster_character(graph, grouping, cut_edges, hw_max_qubit=24):
    K = graph.edge_count
    max_d = 0
    cumulative_hardness = 0.0
    for idx, group in enumerate(grouping):
        print('group is:', group)
        group_K = 0
        group_d = 0
        for vertex in group.split(';'):
            print('looking at vertex:', vertex)
            qargs = vertex.split(' ')
            for qarg in qargs:
                if int(qarg.split(']')[1]) == 0:
                    print('qarg %s is a starting node, d++'%qarg)
                    group_d += 1
            for u, v in cut_edges:
                if vertex == v:
                    print('vertex %s is cutting dest node, d++, K++'%vertex)
                    group_K += 1
                    group_d += 1
                elif vertex == u:
                    print('vertex %s is cutting src node, K++'%vertex)
                    group_K += 1
        print('K = %d, d = %d' % (group_K, group_d))
        cumulative_hardness += float('inf') if group_d > hw_max_qubit else np.power(2,group_d)*np.power(8,group_K)
        max_d = max(max_d, group_d)
    print('cumulative hardness =', cumulative_hardness)
    print('-'*100)
    return K, max_d, cumulative_hardness

if __name__ == '__main__':
    circ = gen_supremacy(3,4,8,'71230456')
    stripped_circ = circ_stripping(circ)
    graph = circuit_to_graph(stripped_circ)
    positions, hardness, K, d = min_cut(graph, 2, 24)