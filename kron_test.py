import numpy as np
from scipy import sparse
from scipy.sparse import random, kron, sputils
from time import time
from numpy.linalg import norm

arr_a = np.random.rand(2**2)
arr_b = np.random.rand(2**26)
begin = time()
np.kron(arr_a,arr_b)
print('%.3f seconds'%(time()-begin))

begin = time()
mid = int(len(arr_a)/2)
np.kron(arr_a[:mid],arr_b)
np.kron(arr_a[mid:],arr_b)
print('%.3f seconds'%(time()-begin))