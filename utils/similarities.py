from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from math import sqrt
import itertools


def cosine_sim(dicts, x1, x2, min_support=5):
    prod = 0
    denom1 = 0
    denom2 = 0
    num = 0
    for i in dicts[x1]:
        if i in dicts[x2]:
            num += 1
            prod += dicts[x1][i] * dicts[x2][i]
            denom1 += dicts[x1][i] ** 2
            denom2 += dicts[x2][i] ** 2

    if num < min_support:
        return 0
    try:
        return prod / sqrt(denom1 * denom2)
    except ZeroDivisionError:
        return 0


def msd_sim(dicts, x1, x2, min_support=5):
    sq_diff = 0
    num = 0
    for i in dicts[x1]:
        if i in dicts[x2]:
            num += 1
            sq_diff = (dicts[x1][i] - dicts[x2][i]) ** 2

    if num < min_support:
        return 0
    return 1 / (sq_diff / num + 1)


def pearson_sim(dicts, x1, x2, min_support=5, shrunk=10):
    prods = 0
    num = 0
    sqi = 0
    sqj = 0
    si = 0
    sj = 0
    for i in dicts[x1]:
        if i in dicts[x2]:
            num += 1
            prods += dicts[x1][i] * dicts[x2][i]
            sqi += dicts[x1][i] ** 2
            sqj += dicts[x2][i] ** 2
            si += dicts[x1][i]
            sj += dicts[x2][i]

    if num < min_support:
        return 0
    if shrunk:
        n_shrunk = num / (num + shrunk)
    else:
        n_shrunk = 1.0

    denum = sqrt((num * sqi - si ** 2) * (num * sqj - sj ** 2))
    if denum == 0:
        return 0
    else:
        return n_shrunk * (num * prods - si * sj) / denum


def get_sim(data, sim_func, n, ids, symmetric=True, min_support=5):
    if not symmetric:
        print("not symmetric")
        sim = np.array([[
            sim_func(data, i, other, min_support) for other in ids] for i in ids])
    else:
        print("symmetric")
        sim = np.zeros((n, n))
        for i in ids:
            sim[i, i] = 1.0
            for j in ids[i+1: ]:
                sim[i, j] = sim_func(data, i, j, min_support)
                sim[j, i] = sim[i, j]
    return sim


def map_func(dataset, sim_func, i, train_ids, min_support=5):
    return [sim_func(dataset, i, other, min_support) for other in train_ids]


def get_sim_parallel(data, sim_func, i, ids, n_jobs=4, min_support=5):
    params = [[data], [sim_func], i, [ids], [min_support]]
    with Pool(processes=n_jobs) as p:
        sim = p.starmap(map_func, itertools.product(*params))
        sim = np.array(sim)
    return sim





