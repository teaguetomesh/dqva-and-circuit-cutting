import multiprocessing
import time
import os
import numpy as np
def f(x,y):
    time.sleep(max(x,y))
    return max(x,y)

if __name__ == '__main__':
    data_size = 5
    data = np.random.rand(data_size,2)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    serial_begin = time.time()
    results = [f(*x) for x in data]
    serial_time = time.time()-serial_begin
    print('serial_time = %f'%serial_time)
    print(results)

    pool_begin = time.time()
    results = pool.starmap(f,data)
    pool_time = time.time()-pool_begin
    print('pool_time = %f'%pool_time)
    print(results)