#!/usr/bin/env python3.5
# coding: utf-8

from __future__ import print_function, division
import argparse
from pareto_models import Bnc_Pareto, Basic_Pareto

def read_graph(graph_fname, cons_fname=None):
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
    parser.add_argument('method', choices=['basic', 'bnc'], 
                        help='mip model to be used')
    parser.add_argument('--lower', type=int, help='lower bound for minsize',
                        default=None)
    parser.add_argument('--upper', type=int, help='upper bound for minsize',
                        default=None)
    parser.add_argument('--verbose', action='store_true', 
                        help='print detailed output')
    parser.add_argument('--overlap', action='store_true',
                        help='allow overlap between clusters')
    args = parser.parse_args()    

    n_vertices, edges, constraints, node_ids = read_graph(args.graph_file, 
                                                          args.cons_file)
    verbosity = 1 if args.verbose else 0
    overlap = args.overlap
    k = args.k
    
    lower = args.lower if args.lower is not None else 0
    upper = args.upper if args.upper is not None else n_vertices // k
    
    kwargs = dict(n_vertices=n_vertices, 
                  edges=edges, 
                  constraints=constraints, 
                  k=k,
                  verbosity=verbosity, 
                  overlap=overlap)
    
    if args.method == 'bnc':
        Model = Bnc_Pareto
    elif args.method == 'basic':
        Model = Basic_Pareto
    
    print('number of nodes:%d'%n_vertices)
    print('number of clusters:%d'%k)
    print('largest possible minsize bound:%d\n'%(n_vertices//k))    
    
    
    print('bound,size,violations')

    minsize_bound = 0
    done = False
    while not done:
        kwargs['minsize'] = minsize_bound
        m = Model(**kwargs)
        m.solve()
        
        if not m.optimal:
            print('%d,-,-'%minsize_bound)
            done = True
        else: 
            minsize = min(len(cluster) for cluster in m.clusters)
            print('%d,%d,%f'%(minsize_bound, minsize, m.objective))
            minsize_bound = minsize + 1

