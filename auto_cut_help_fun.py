from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import random
import math
import copy
from collections import Counter
from datetime import datetime
import sys

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

def translate(original_r, original_g, curr_g, grouping):
    original_head, original_tail = original_g.edges[original_r]
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
        g_r = translate(original_r=graph_r, original_g=graph, curr_g=g, grouping=grouping)
        g_head, g_tail = g.edges[g_r]
        print('contracting', graph.edges[graph_r])

        # print('keep the edge', (head, tail))
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

    cut_edges = []
    for edge in graph.edges:
        if edge not in contracted_edges:
            cut_edges.append(edge)
            print('original edge', edge)
            print('is not contracted in', contracted_edges)
            print()
    return g, grouping, cut_edges

def cluster_character(grouping):
    # FIXME: double check d calculation
    max_d = 0
    for group in grouping:
        d = 0
        for vertex in group.split(' '):
            qargs = vertex.split(',')
            for qarg in qargs:
                if qarg[len(qarg)-1] == '0':
                    d += 1
        max_d = max(max_d, d)
    return max_d

# Karger's Algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nC2 logn times
def min_cut(graph):
    m = graph.edge_count
    n = graph.vertex_count
    all_K_d = {}
    print('will run %d times' % int(n * (n-1) * math.log(n)/2))
    for i in range(int(n * (n-1) * math.log(n)/2)):
        random.seed(datetime.now())
        g, grouping, cut_edges = contract(graph)
        m = min(m, g.edge_count)
        K = g.edge_count
        d = cluster_character(grouping)
        # TODO: K,d cluster is overwritten
        all_K_d[(K,d)] = cut_edges
        # print('Run %d cut_edges:' % i, cut_edges)
        # print('*'* 100)
    
    pareto_K_d = {}
    for key0 in all_K_d:
        K0, d0 = key0
        is_pareto = True
        for key1 in all_K_d:
            K1, d1 = key1
            if key1!=key0 and ((K1<K0 and d1<d0) or (K1==K0 and d1<d0) or (K1<K0 and d1==d0)):
                is_pareto = False
                break
        if is_pareto:
            pareto_K_d[key0] = all_K_d[key0]

    return pareto_K_d

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

def find_pareto_solutions(circ):
    stripped_circ = circ_stripping(circ)
    graph = circuit_to_graph(stripped_circ)
    pareto_K_d = min_cut(graph)
    return pareto_K_d