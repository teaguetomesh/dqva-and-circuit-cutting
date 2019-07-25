#!/usr/bin/env python3.5
# coding: utf-8

from __future__ import print_function, division
import argparse
from bnc_model import Bnc_Model
from basic_model import Basic_Model

def read_graph(graph_fname, cons_fname=None):
    node_names = dict()
    node_ids = dict()
    edges = set()
    
    def get_id(name):
        # node_id is ordered 0 to n_vertices
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
    
    
    constraints = []
    if cons_fname is not None:
        with open(cons_fname) as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line: continue
                [n1, n2, w] = line.split(',')
                [v1, v2] = [get_id(i) for i in (n1, n2)]
                w = float(w)
                constraints.append((v1, v2, w))
    
    return n_vertices, list(edges), constraints, node_ids


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file', help='file containing the edges')
    parser.add_argument('cons_file', help='file containing the constraints')
    parser.add_argument('k', type=int, help='number of clusters')
    parser.add_argument('gamma', type=float, help='balance coefficient')
    parser.add_argument('method', choices=['basic', 'bnc'], 
                        help='mip model to be used')
    parser.add_argument('--timeout', type=float, default=None,
                        help='set a time limit')
    parser.add_argument('--verbose', action='store_true', 
                        help='print detailed output')
    parser.add_argument('--nosym', action='store_true',
                        help='disable symmetry breaking')
    parser.add_argument('--overlap', action='store_true',
                        help='allow overlap between clusters')
    parser.add_argument('--single', action='store_true',
                        help='add at most one cut per cluster')
    args = parser.parse_args()    

    n_vertices, edges, constraints, node_ids = read_graph(args.graph_file, 
                                                          args.cons_file)
    verbosity = 1 if args.verbose else 0
    overlap = args.overlap
    sym = not args.nosym
    k = args.k
    gamma = args.gamma
    timeout = args.timeout
    single_cut = args.single
    
    kwargs = dict(n_vertices=n_vertices, 
                  edges=edges,
                  constraints=constraints,
                  k=k, gamma=gamma,
                  verbosity=verbosity,
                  symmetry_breaking=sym,
                  overlap=overlap,
                  timeout=timeout)

    [print(x, kwargs[x]) for x in kwargs]
    print('*'*100)
    
    if args.method == 'bnc':
        kwargs['single_cut'] = single_cut
        m = Bnc_Model(**kwargs)
    elif args.method == 'basic':
        m = Basic_Model(**kwargs)
        
    m.solve()
    # m.print_stat()
    # print('*'*100)
    print('clusters:')
    print(m.clusters)

    print('node count:', m.node_count)
    print('mip gap:', m.mip_gap)
    print('objective value:', m.objective)
    print('runtime:', m.runtime)

    if (m.optimal):
        print('OPTIMAL')
    else:
        print('NOT OPTIMAL')

