#!/bin/bash

# echo "Using the basic method with symmetry-breaking:"
# python runner.py ../data/graph.csv ../data/graph_cannot_links.csv 3 5 basic

# echo "Using the basic method with overlap:"
# ./runner.py ../data/graph.csv ../data/graph_cannot_links.csv 3 5 basic --verbose --overlap

echo "Using the branch-and-cut method without symmetry-breaking:"
python runner.py ../data/graph.csv ../data/graph_cannot_links.csv 3 5 bnc --verbose --nosym --timeout 10

# echo "Using the branch-and-cut method with overlap:"
# ./runner.py ../data/graph.csv ../data/graph_cannot_links.csv 3 5 bnc --verbose --overlap --timeout 10



