#!/usr/bin/env python3.5
# coding: utf-8

from __future__ import print_function, division
import argparse
import networkx as nx

def create_graph(n_vertices, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    for v1, v2 in edges:
        G.add_edge(v1, v2) 

    return G

def read_graph(graph_fname):
    node_names = dict()
    node_ids = dict()
    edges = set()
    
    def get_id(name):
        if not name in node_names:
            node_id = len(node_names)
            node_names[name] = node_id
            node_ids[node_id] = name 

        return node_names[name]  

    def add_edge(v1, v2):
        first = min(v1, v2)
        second = max(v1, v2)
        edges.add((first, second))



    with open(graph_fname) as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            [n1, n2] = line.strip().split(',')
            [v1, v2] = [get_id(i) for i in (n1, n2)]
            add_edge(v1, v2)

    n_vertices = max(node_ids.keys()) + 1
    return n_vertices, list(edges)

def count_paths(G, edges):
    num_paths = 0
    for v1, v2 in edges:
        for _ in nx.all_simple_paths(G, v1, v2):
            num_paths += 1
    return num_paths
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file', help='file containing the edges')
    args = parser.parse_args()    

    n_vertices, edges = read_graph(args.graph_file)
    G = create_graph(n_vertices, edges)
    num_paths = count_paths(G, edges)
    
    print(args.graph_file, num_paths, n_vertices, len(edges))