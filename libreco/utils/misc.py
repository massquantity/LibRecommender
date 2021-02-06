import functools
import time
from contextlib import contextmanager
from itertools import chain
import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()


def shuffle_data(length, *args):
    mask = np.random.permutation(range(length))
    return tuple(map(lambda x: x[mask], [*args]))


def count_params():
    total_params = np.sum(
        [
            np.prod(v.get_shape().as_list())
            for v in tf.trainable_variables()
        ]
    )
    embedding_params = np.sum(
        [
            np.prod(v.get_shape().as_list())
            for v in tf.trainable_variables()
            if (
                'feat' in v.name
                or 'weight' in v.name
                or 'bias' in v.name
                or 'embed' in v.name
            )
        ]
    )
    network_params = total_params - embedding_params
    total_params = f"{total_params:,}"
    embedding_params = f"{embedding_params:,}"
    network_params = f"{network_params:,}"
    print_params = (f"total params: "
                    f"{colorize(total_params, 'yellow')} | " 
                    f"embedding params: "
                    f"{colorize(embedding_params, 'yellow')} | " 
                    f"network params: "
                    f"{colorize(network_params, 'yellow')}")
    print(print_params)


def assign_oov_vector(model, add=True):
    for v_name in chain.from_iterable(
            [model.user_variables_np, model.item_variables_np]
    ):
        if v_name not in model.__dict__:
            raise KeyError(f"{v_name} is not an attribute of the model.")
        var = model.__dict__[v_name]
        if var.ndim == 1:
            var = np.append(var, np.mean(var))
        else:
            var = np.vstack([var, np.mean(var, axis=0)])
        setattr(model, v_name, var)


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


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson

    Original source from openAI `gym`:
    https://github.com/openai/gym/blob/master/gym/utils/colorize.py
    """

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


