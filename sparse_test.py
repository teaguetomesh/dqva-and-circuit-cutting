import numpy as np
from scipy.sparse import random, kron, sputils
from time import time
from numpy.linalg import norm

np.random.seed(10)

sparse_A = random(1, 2**14, format='csr', density=0.1)
sparse_B = random(1, 2**14, format='csr', density=0.1)
sparse_C = random(1, 2**14, format='csr', density=0.1)

begin = time()
scipy_kron = kron(sparse_A,sparse_B)
scipy_kron = kron(scipy_kron,sparse_C)
print('sparse took %.2f seconds'%(time()-begin))

# dense_A = sparse_A.toarray()
# dense_B = sparse_B.toarray()
# begin = time()
# np_kron = np.kron(dense_A,dense_B)
# print('np took %.2f seconds'%(time()-begin))

# print(type(scipy_kron),scipy_kron.shape)
# print(type(np_kron),np_kron.shape)

# print(norm(scipy_kron-np_kron))