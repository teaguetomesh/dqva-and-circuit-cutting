from __future__ import print_function
from __future__ import division

import random
import math
import copy
from collections import Counter
from datetime import datetime

# Implement contraction using Counter objects
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
        hi, ti = self.edges[edge_index]
        # print('removing edge ', self.edges[edge_index])
        
        head = self.verts[hi]
        tail = self.verts[ti]

        # print('head = ', head, 'tail = ', tail)

        # Remove the edge between head and tail
        del head[ti]
        del tail[hi]

        # print('After removing head->tail, head = ', head, 'tail = ', tail)

        # Merge tails
        head.update(tail)

        # print('After merging tail, head = ', head)

        # Update all the neighboring vertices of the fused vertex
        for i in tail.keys():
            v = self.verts[i]
            v[hi] += v[ti]
            del v[ti]
            
        # Finally remove the tail vertex
        del self.verts[ti]

        self.update_edges()
        
def contract(graph, min_v=2):
    g = copy.deepcopy(graph)
    while g.vertex_count > min_v:
        r = random.randrange(0, g.edge_count)
        g.merge_vertices(r)

    return g

# Karger's Algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nC2 logn times
def min_cut(graph):
    m = graph.edge_count
    n = graph.vertex_count
    for i in range(int(n * (n-1) * math.log(n)/2)):
        random.seed(datetime.now())
        g = contract(graph)
        m = min(m, g.edge_count)
        # print(i, m)
    return m

def _fast_min_cut(graph):
    if graph.vertex_count <= 6:
        return min_cut(graph)
    else:
        t = math.floor(1 + graph.vertex_count / math.sqrt(2))
        g1 = contract(graph, t)
        g2 = contract(graph, t)
        
        # print('m = ', min(_fast_min_cut(g1), _fast_min_cut(g2)))
        return min(_fast_min_cut(g1), _fast_min_cut(g2))

# Karger Stein algorithm. Refer https://en.wikipedia.org/wiki/Karger%27s_algorithm#Karger.E2.80.93Stein_algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nlogn/(n - 1) times
def fast_min_cut(graph):
    m = graph.edge_count
    n = graph.vertex_count
    print('will run %d times' % int(n * math.log(n)/(n - 1)))
    for i in range(int(n * math.log(n)/(n - 1))):
        random.seed(datetime.now())
        m = min(m, _fast_min_cut(graph))
        # print(i, m)
    return m
    
# Simple test
# graph = Graph([[4,6,7], [5,6,7], [6,4,5], [7,4,5]])
# graph = Graph([['a','c','d'],['b','c','d'],['c','a','b'],['d','a','b']])
graph = Graph([['a','c','d'],['b','c','d'],['c','a','b'],['d','a','b']])
print(graph.verts)
print(graph.edges)
print('*' * 100)
print(fast_min_cut(graph))