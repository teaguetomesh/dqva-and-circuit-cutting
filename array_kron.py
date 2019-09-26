import numpy as np
from time import time
import itertools

if __name__ == '__main__':
    num_elements = 10000
    arr_a = np.random.rand(num_elements)
    arr_b = np.random.rand(num_elements)
    
    begin = time()
    arr_c = np.kron(arr_a,arr_b)
    print('Python kronecker time elapsed = %f seconds'%(time()-begin))

    combinations = itertools.product(range(1,9),repeat=6)
    begin = time()
    for s in combinations:
        1+1
    print('Python combinations iteration time elapsed = %f seconds'%(time()-begin))