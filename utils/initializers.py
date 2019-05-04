import numpy as np


def truncated_normal_slow(mean, scale, shape):
    total_num = np.multiply(*shape)
    data = []
    while len(data) != total_num:
        sample = np.random.normal(mean, scale, 1)
        while sample > mean + 2 * scale or sample < mean - 2 * scale:
            sample = np.random.normal(mean, scale, 1)
        data.append(sample)
    array = np.array(data).reshape(*shape)
    return array


def truncated_normal(shape, mean=0.0, scale=0.05):
    total_num = np.multiply(*shape)
    array = np.random.normal(mean, scale, total_num)
    for i, sample in enumerate(array):
        while sample > mean + 2 * scale or sample < mean - 2 * scale:
            sample = np.random.normal(mean, scale, 1)
            array[i] = sample
    return array.reshape(*shape)


def xavier_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return truncated_normal(mean=0.0, scale=std, shape=[fan_in, fan_out])


def he_init(fan_in, fan_out):
    std = 2.0 / np.sqrt(fan_in + fan_out)
    # std = np.sqrt(2.0 / fan_in)
    return truncated_normal(mean=0.0, scale=std, shape=[fan_in, fan_out])


def variance_scaling(scala, fan_in=None, fan_out=None, mode="fan_in"):
    """
    xavier:  mode = "fan_average", scale = 1.0
    he: mode = "fan_in", scale = 2.0
    he2: mode = "fan_average", scale = 2.0
    """
    if mode == "fan_in":
        std = np.sqrt(scala / fan_in)
    elif mode == "fan_out":
        std = np.sqrt(scala / fan_out)
    elif mode == "fan_average":
        std = np.sqrt(2.0 * scala / (fan_in + fan_out))
    else:
        raise ValueError("mode must be one of these: fan_in, fan_out, fan_average")
    return truncated_normal(mean=0.0, scale=std, shape=[fan_in, fan_out])



