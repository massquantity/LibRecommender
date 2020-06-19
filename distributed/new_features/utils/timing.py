import time
import functools
from contextlib import contextmanager


def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} elapsed: {(end - start):3.3f}s")
        return result
    return wrapper


@contextmanager
def time_block(block_name="block", verbose=1):
    if verbose > 0:
        start = time.perf_counter()
        try:
            yield
        except Exception:
            raise
        else:
            end = time.perf_counter()
            print(f"{block_name} elapsed: {(end - start):3.3f}s")

    else:
        try:
            yield
        except Exception:
            raise
