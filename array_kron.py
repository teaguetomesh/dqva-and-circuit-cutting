import numpy as np
from time import time

if __name__ == '__main__':
    num_elements = 10000
    arr_a = np.random.rand(num_elements)
    arr_b = np.random.rand(num_elements)
    begin = time()
    for i in range(10):
        arr_c = np.kron(arr_a,arr_b)
    print('Python time elapsed = %f seconds'%(time()-begin))