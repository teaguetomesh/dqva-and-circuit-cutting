import glob
import networkx as nx

dirs = []
for n in range(26, 27):
    temp_dirs = glob.glob('N{}*'.format(n))
    dirs.extend(temp_dirs)
print('Dirs:', dirs)

for folder in dirs:
    n = int(folder.split('_')[0][1:])
    d = int(folder.split('_')[1][1:])
    print('Nodes: {}, degree: {}'.format(n, d))

    for j in range(15):
        G = nx.random_regular_graph(d, n)
        edges = list(G.edges())

        with open(folder+'/G{}.txt'.format(j+1), 'w') as fn:
            edgestr = ''.join(['{}, '.format(e) for e in edges])
            edgestr = edgestr.strip(', ')
            fn.write(edgestr)

