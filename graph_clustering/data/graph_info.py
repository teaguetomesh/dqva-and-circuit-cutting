import networkx
import csv
import glob
import shutil

def edgeFileInfo(edgeFile):
    with open(edgeFile, 'r') as f:
        reader = csv.reader(f)
        edgeList = list(reader)
    lines = []
    for edge in edgeList[1:]:
        lines.append(edge[0] + ',' + edge[1])
    G = networkx.parse_edgelist(lines, delimiter=",")
    
    if networkx.number_of_nodes(G) < 100:
        print('-----')
        print(edgeFile)
#        shutil.copy(edgeFile, 'graphs_less_than_100_nodes')
        print('average_clustering_coefficient: ' + str(networkx.average_clustering(G)))
        print('transitivity: ' + str(networkx.transitivity(G)))
        print('node_connectivity: ' + str(networkx.node_connectivity(G)))
        print('node count: ' + str(networkx.number_of_nodes(G)))
        print('edge count: ' + str(networkx.number_of_edges(G)))

def graphsInfo(filesRegexp):
    for filename in glob.iglob(filesRegexp):
        edgeFileInfo(filename)

def main():
    graphsInfo('graphs_less_than_100_nodes/*graph')

main()
