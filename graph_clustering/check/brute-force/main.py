
# coding: utf-8
import random
import networkx as nx
from bfcheck import BF_Checker

def get_graph(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

connected = False
while not connected:
    G = nx.gnm_random_graph(13, 20)
    connected = nx.is_connected(G)

n = len(G.nodes())
edges = G.edges()
g = get_graph(n, edges)


constraints = []
for i in range(10):
    first = random.randint(0, n-1)
    second = random.randint(0, n-1)
    if first != second:
        weight = random.random()
        constraints.append([first, second, weight])
        

k = 2
gamma = random.random() * 10

bfc = BF_Checker(g, constraints, k, gamma)
print(bfc.is_feasible())
print(bfc.get_best())
