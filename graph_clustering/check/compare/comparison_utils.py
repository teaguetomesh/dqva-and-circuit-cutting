import sys
sys.path.append('../../clustering_model')
sys.path.append('../brute-force')

from bfcheck import BF_Checker
from clustering_model import Clustering_Model

def get_canonical(in_edges):
    node_dict = dict()
    def get_id(node):
        if node not in node_dict:
            node_dict[node] = len(node_dict)
        return node_dict[node]
    
    g = dict()
    def add_edge(u, v):
        if u not in g:
            g[u] = set()
        g[u].add(v)
    
    def add_both(u, v):
        add_edge(u, v)
        add_edge(v, u)
        
    for edge in in_edges:
        [x, y] = [get_id(u) for u in edge]
        add_both(x, y)
        
    n = len(g)
    edges = []
    graph = [[] for _ in range(n)]
    for u in g:
        for v in g[u]:
            if (u < v):
                graph[u].append(v)
                graph[v].append(u)
                edges.append((u, v))
                
    return n, edges, graph
    
def compare(edges, constraints, k, gamma, verbosity=0):
    n, edges, graph = get_canonical(edges)
    bfc = BF_Checker(graph, constraints, k, gamma)
    bfc_opt = bfc.is_feasible()
    bfc_obj = bfc.get_best()
    
    cm = Clustering_Model(n, edges, constraints, k, gamma)
    cm.solve()
    cm_opt = cm.optimal
    cm_obj = cm.objective

    return bfc_opt == cm_opt and abs(bfc_obj-cm_obj)<1e-6
