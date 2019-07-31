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
        self.verts = {}
        all_vertices = []
        for l in vlist:
            [all_vertices.append(x) for x in l]
            v = l[0]
            vertex = Counter(l[1:])
            self.verts[v] = vertex
        all_vertices = set(all_vertices)
        for v in all_vertices:
            if v not in self.verts:
                self.verts[v] = Counter([])
        self.update_edges()

    def update_edges(self):
        self.edges = []
        
        for src in self.verts:
            dest_l = list(self.verts[src].elements())
            self.edges += [(src,x) for x in dest_l]

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        head_key, tail_key = self.edges[edge_index]
        
        head = self.verts[head_key]
        tail = self.verts[tail_key]

        # Remove the edge from head to tail
        del head[tail_key]

        # Merge tail into head
        head.update(tail)

        # Vertices pointing to tail should now point to head
        for vertex in self.verts:
            if tail_key in self.verts[vertex] and vertex != head_key:
                if head_key in self.verts[vertex]:
                    self.verts[vertex][head_key] += self.verts[vertex][tail_key]
                else:
                    self.verts[vertex][head_key] = self.verts[vertex][tail_key]
                del self.verts[vertex][tail_key]
            
        # Finally remove the tail vertex
        del self.verts[tail_key]

        self.update_edges()

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def circuit_to_graph(circ):
    dag = circuit_to_dag(circ)
    input_qubit_gate_counter = {}
    for qubit in circ.qubits:
        input_qubit_gate_counter[qubit] = 0
    
    id_name = {}
    for vertex in dag.topological_op_nodes():
        vertex_name = ''
        for qarg in vertex.qargs:
            gate_count = input_qubit_gate_counter[qarg]
            vertex_name += '%s[%d]%d ' % (qarg[0].name, qarg[1], gate_count)
            input_qubit_gate_counter[qarg] += 1
        vertex_name = vertex_name[:len(vertex_name)-1]
        id_name[id(vertex)] = vertex_name
    
    graph_edges = {}
    for edge in dag.edges():
        source, dest, attr = edge
        if source.type == 'op' and dest.type == 'op':
            source_vertex_name = id_name[id(source)]
            dest_vertex_name = id_name[id(dest)]
            # print(source_vertex_name, 'to', dest_vertex_name)
            if source_vertex_name in graph_edges:
                graph_edges[source_vertex_name].append(dest_vertex_name)
            else:
                graph_edges[source_vertex_name] = [dest_vertex_name]
    
    # [print(x, graph_edges[x]) for x in graph_edges]

    graph_init = []
    for source_vertex in graph_edges:
        graph_init_ele = [source_vertex]
        for dest_vertex in graph_edges[source_vertex]:
            graph_init_ele.append(dest_vertex)
        graph_init.append(graph_init_ele)
    # [print(x) for x in graph_init]

    graph = Graph(graph_init)
    return graph

def translate_idx(graph_edge, grouping, g):
    graph_edge_head, graph_edge_tail = graph_edge
    g_edge_head = None
    g_edge_tail = None
    hi = None
    ti = None
    g_contraction_idx = None
    for idx, group in enumerate(grouping):
        if graph_edge_head in group:
            g_edge_head = group[0]
            hi = idx
        if graph_edge_tail in group:
            g_edge_tail = group[0]
            ti = idx
    # print('graph_edge_head is in group', grouping[hi], 'converting to ', g_edge_head)
    # print('graph_edge_head is in group', grouping[ti], 'converting to ', g_edge_tail)
    if hi!=ti and hi!=None and ti!=None:
        grouping[hi] += grouping[ti]
        del grouping[ti]
        for idx, g_edge in enumerate(g.edges):
            if (g_edge_head, g_edge_tail) == g_edge:
                g_contraction_idx = idx
                # print('contracting edge', g.edges[g_contraction_idx], 'in g')
    return g_contraction_idx, grouping

def contract(graph, min_v=2):
    g = copy.deepcopy(graph)
    grouping = [[x] for x in g.verts]
    contraction_order = random.sample(range(0,graph.edge_count), graph.edge_count)
    counter = 0
    while g.vertex_count > min_v:
        graph_edge_idx = contraction_order[counter]
        graph_edge = graph.edges[graph_edge_idx]
        # print('contracting edge', graph_edge, 'in graph')
        g_edge_idx, grouping = translate_idx(graph_edge, grouping, g)
        if g_edge_idx != None:
            # print('contracting edge', g.edges[g_edge_idx], 'in g')
            g.merge_vertices(g_edge_idx)
        # print(len(g.edges))
        counter += 1
    # print('contracted %d/%d edges' %(counter, len(contraction_order)))
    cut_edges = [graph.edges[x] for x in contraction_order[counter:]]
    # print('original edges not being contracted:', cut_edges)
    true_cut_edges = []
    for edge in cut_edges:
        implicitly_contracted = False
        u, v = edge
        # print('edge', edge, 'is not explicitly contracted')
        for group in grouping:
            if u in group and v in group:
                implicitly_contracted = True
                # print('but is implicitly contracted')
                break
        if not implicitly_contracted:
            true_cut_edges.append(edge)
    # print('cut_edges:', true_cut_edges)
    # print('grouping is:', grouping)
    # print('remaining edges in g:', g.edges)
    return g, grouping, true_cut_edges

def cluster_character(grouping, cut_edges, hw_max_qubit=24):
    d = []
    K = []
    cumulative_hardness = 0.0
    for idx, group in enumerate(grouping):
        # print('group is:', group)
        group_K = 0
        group_d = 0
        for vertex in group:
            # print('looking at vertex:', vertex)
            qargs = vertex.split(' ')
            for qarg in qargs:
                if int(qarg.split(']')[1]) == 0:
                    # print('qarg %s is a starting node, d++'%qarg)
                    group_d += 1
            for u, v in cut_edges:
                if vertex == v:
                    # print('vertex %s is cutting dest node, d++, K++'%vertex)
                    group_K += 1
                    group_d += 1
                elif vertex == u:
                    # print('vertex %s is cutting src node, K++'%vertex)
                    group_K += 1
        # print('K = %d, d = %d' % (group_K, group_d))
        K.append(group_K)
        d.append(group_d)
        if group_d>hw_max_qubit:
            cumulative_hardness = float('inf')
            break
        else:
            # TODO: better ways to catch overflow?
            # Exponent divided by 10 to prevent overflow
            cumulative_hardness += np.power(2,(group_d+3*group_K)/10)
    return K, d, cumulative_hardness

def min_cut(graph, min_v=2, hw_max_qubit=20):
    min_hardness = float('inf')
    min_hardness_cuts = None
    min_hardness_K = None
    min_hardness_d = None
    for trial in range(2000):
        random.seed(datetime.now())
        g, grouping, cut_edges = contract(graph, min_v)
        K, d, hardness = cluster_character(grouping, cut_edges, hw_max_qubit)
        if hardness<min_hardness:
            min_hardness = hardness
            min_hardness_cuts = cut_edges
            min_hardness_K = K
            min_hardness_d = d
    return min_hardness_cuts, min_hardness, min_hardness_K, min_hardness_d

def positions_parser(stripped_circ_cuts, circ):
    dag = circuit_to_dag(circ)
    circ_cuts = []
    for cut in stripped_circ_cuts:
        multiQ_gate_idx = None
        wire = None
        # print('cut in stripped circ:', cut)
        u, v = cut
        u_qargs = [x.split(']')[0]+']' for x in u.split(' ')]
        v_qargs = [x.split(']')[0]+']' for x in v.split(' ')]
        common_vertex = list(set(u_qargs).intersection(v_qargs))
        # FIXME: how to account for this possibility in the input circuit?
        if len(common_vertex)>1:
            raise Exception('vertices have more than one edge in between')
        common_vertex = common_vertex[0]
        for qarg in u.split(' '):
            if qarg.split(']')[0]+']' == common_vertex:
                multiQ_gate_idx = int(qarg.split(']')[1])
        for qubit in circ.qubits:
            if '%s[%d]' % (qubit[0].name,qubit[1]) == common_vertex:
                wire = qubit
        # print('wire:', wire, 'multiQ_gate_idx=', multiQ_gate_idx)
        counter = 0
        for idx, op_node in enumerate(dag.nodes_on_wire(wire, only_ops=True)):
            # TODO: only counting 2q gates now
            if len(op_node.qargs) == 2:
                if counter == multiQ_gate_idx:
                    circ_cuts.append((wire, idx))
                    break
                else:
                    counter += 1
    return circ_cuts

if __name__ == '__main__':
    circ = gen_supremacy(5,5,8,'71230456')
    stripped_circ = circ_stripping(circ)
    graph = circuit_to_graph(stripped_circ)
    positions, hardness, K, d = min_cut(graph, 3, 14)
    if hardness == float('inf'):
        raise Exception('cannot find any cut')
    positions = positions_parser(positions, circ)
    print('%d cuts at:'%len(positions), positions)
    [print('cluster %d, K ='%i,K[i],'d =',d[i]) for i in range(len(K))]
    print('objective =', hardness)
    print('*'*100)