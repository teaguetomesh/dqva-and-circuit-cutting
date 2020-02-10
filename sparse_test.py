import numpy as np
from scipy import sparse
from scipy.sparse import random, kron, sputils
from time import time
from numpy.linalg import norm

np.random.seed(10)

sizes = [24,24,6]
sparse_matrices = []
for i in sizes:
    i = int(i/2)
    sparse_matrix = random(2**i, 2**i, format='csr', density=0.1)
    sparse_matrices.append(sparse_matrix)

begin = time()
scipy_kron = kron(sparse_matrices[0],sparse_matrices[1],format='csr')
print('sparse took %.2f seconds'%(time()-begin))

begin = time()
scipy_kron = kron(scipy_kron,sparse_matrices[2],format='csr')
print('sparse took %.2f seconds'%(time()-begin))