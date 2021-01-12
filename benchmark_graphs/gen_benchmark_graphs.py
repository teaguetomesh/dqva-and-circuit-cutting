import glob
import networkx as nx


dirs = glob.glob('N*')

for folder in dirs:
    n = int(folder.split('_')[0][1:])
    d = int(folder.split('_')[1][1:])
    print('Nodes: {}, degree: {}'.format(n, d))

    for j in range(100):
        G = nx.random_regular_graph(d, n)
        edges = list(G.edges())

        with open(folder+'/G{}.txt'.format(j+1), 'w') as fn:
            edgestr = ''.join(['{}, '.format(e) for e in edges])
            edgestr = edgestr.strip(', ')
            fn.write(edgestr)

