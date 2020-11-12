# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import multiprocessing
from time import time
import os
import numpy as np

def f(x):
    return x*x

if __name__ == '__main__':
    data_size = 500000
    data = np.random.rand(data_size)
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size)
    chuck_size = data_size // pool_size // 10
    print("chuck size is", chuck_size)

    serial_begin = time()
    result = [f(x) for x in data]
    serial_time = time()-serial_begin
    print('serial_time = %f'%serial_time)
    print(result[:3])

    pool_begin = time()
    result = pool.imap(f,data, chunksize=chuck_size)
    pool_time = time()-pool_begin
    print('pool_time = %f'%pool_time)
    [print(next(result)) for x in range(3)]