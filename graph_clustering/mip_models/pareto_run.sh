#!/bin/bash

echo "Pareto front obtained by the basic method without overlap:"
./pareto_runner.py ../data/graph.csv ../data/graph_cannot_links.csv 3 basic 

echo "Pareto front obtained by the branch-and-cut method with overlap:"
./pareto_runner.py ../data/graph.csv ../data/graph_cannot_links.csv 3 bnc --overlap



