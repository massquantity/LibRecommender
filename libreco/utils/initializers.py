import numpy as np


def truncated_normal(
    np_rng: np.random.Generator,
    shape,
    mean=0.0,
    scale=0.05,
    tolerance=5,
):
    # total_num = np.multiply(*shape)
    total_num = shape if len(shape) == 1 else np.multiply(*shape)
    array = np_rng.normal(mean, scale, total_num).astype(np.float32)
    upper_limit, lower_limit = mean + 2 * scale, mean - 2 * scale
    for _ in range(tolerance):
        index = np.logical_or((array > upper_limit), (array < lower_limit))
        num = len(np.where(index)[0])
        if num == 0:
            break
        array[index] = np_rng.normal(mean, scale, num)
    return array.reshape(*shape)


def xavier_init(np_rng, fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return truncated_normal(np_rng, mean=0.0, scale=std, shape=[fan_in, fan_out])


def he_init(np_rng, fan_in, fan_out):
    std = 2.0 / np.sqrt(fan_in + fan_out)
    # std = np.sqrt(2.0 / fan_in)
    return truncated_normal(np_rng, mean=0.0, scale=std, shape=[fan_in, fan_out])


def variance_scaling(np_rng, scale, fan_in=None, fan_out=None, mode="fan_in"):
    """
    xavier:  mode = "fan_average", scale = 1.0
    he: mode = "fan_in", scale = 2.0
    he2: mode = "fan_average", scale = 2.0
    """
    if mode == "fan_in":
        std = np.sqrt(scale / fan_in)
    elif mode == "fan_out":
        std = np.sqrt(scale / fan_out)
    elif mode == "fan_average":
        std = np.sqrt(2.0 * scale / (fan_in + fan_out))
    else:
        raise ValueError("mode must be one of these: fan_in, fan_out, fan_average")
    return truncated_normal(np_rng, mean=0.0, scale=std, shape=[fan_in, fan_out])
