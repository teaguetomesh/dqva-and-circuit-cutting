import functools
from time import process_time

def timeit(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = process_time()
        value = func(*args, **kwargs)
        end_time = process_time()
        run_time = end_time - start_time
        print(f'Finished {func.__name__!r} in {run_time:.4f} seconds')
        return value
    return wrapper_timer