import multiprocessing
import time
import os
import numpy as np

def f(x):
    time.sleep(x)
    return x

if __name__ == '__main__':
    data_size = 10
    data = np.random.rand(data_size)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    serial_begin = time.time()
    [f(x) for x in data]
    serial_time = time.time()-serial_begin
    print('serial_time = %f'%serial_time)

    pool_begin = time.time()
    pool.map(f,data)
    pool_time = time.time()-pool_begin
    print('pool_time = %f'%pool_time)