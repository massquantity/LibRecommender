import time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, issparse
from math import sqrt
import itertools
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


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
            sq_diff += (dicts[x1][i] - dicts[x2][i]) ** 2

    if num < min_support:
        return 0
    try:
        return 1 / (sq_diff / num + 1)
    except ZeroDivisionError:
        return 0


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
        sim = np.array([[sim_func(data, i, other, min_support) for other in ids] for i in ids])
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


def invert_sim(data, n_users, min_support=5):
    prods = np.zeros((n_users, n_users))
    num = np.zeros((n_users, n_users))
    denom1 = np.zeros((n_users, n_users))
    denom2 = np.zeros((n_users, n_users))
    sim = np.zeros((n_users, n_users))

    for i, u_labels in data.items():
        for ui, li in u_labels.items():
            for uj, lj in u_labels.items():
                num[ui, uj] += 1
                prods[ui, uj] += li * lj
                denom1[ui, uj] += li * li
                denom2[ui, uj] += lj * lj

    for ui in range(n_users):
        sim[ui, ui] = 1.0
        for uj in range(ui + 1, n_users):
            if num[ui, uj] < min_support:
                sim[ui, uj] = 0.0
            else:
                denom = np.sqrt(denom1[ui, uj] * denom2[ui, uj])
                try:
                    sim[ui, uj] = prods[ui, uj] / denom
                except ZeroDivisionError:
                    sim[ui, uj] = 0.0
            sim[uj, ui] = sim[ui, uj]
    return sim


try:
    from .similarities_cy import sk_num
except:
    pass

def sk_sim(data, n_users, n_items, min_support=5, sparse=True):
    if sparse:
        user_indices = []
        item_indices = []
        values = []
        for i, u_ratings in data.items():
            for u, r in u_ratings.items():
                user_indices.append(u)
                item_indices.append(i)
                values.append(r)
        m = csr_matrix((np.array(values), (np.array(user_indices), np.array(item_indices))), dtype=np.float32)
        assert issparse(m)
    else:
        m = np.zeros((n_users, n_items))
        for i, u_ratings in data.items():
            for u, r in u_ratings.items():
                m[u, i] = r

    sim = cosine_similarity(m, dense_output=False)
    sim = sim.tolil()

    if min_support > 1:
        print("warning: allow min_support > 1 could be slow for calculating similarity matrix")
        t0 = time.time()
        dot = linear_kernel(m, dense_output=False)
        dot = dot.tolil()
    #    print("dot: ", [i for i in dot.data if len(i) != 0], dot.getnnz())
        s = lil_matrix((n_users, n_users), dtype=np.float32)
        for u in range(n_users):
            for e, freq in enumerate(dot.data[u]):
                index = dot.rows[u][e]
                if freq >= min_support and index != u:
                #    print(u, index, freq)
                    s[u, index] = sim[u, index]
        #    s[u, u] = 1.0
        print("lil time: {:.4f}, lil_elements: {}".format(time.time() - t0, s.getnnz()))
        return s

    if min_support > 0 and not sparse:
        item_user_sklearn = {k: list(v.keys()) for k, v in data.items()}
        num = sk_num(n_users, item_user_sklearn)
        indices = np.where(num < min_support)
        sim[indices[0], indices[1]] = 0.0
        print("diag sim, ", min(sim.diagonal()))
        print("non-zero items: {}, shape: {}".format(sim.getnnz(), sim.shape))

    return sim
