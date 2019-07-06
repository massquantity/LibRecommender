cimport numpy as np
import numpy as np

def cosine_cy(n, data, min_support=5):
    cdef np.ndarray[np.double_t, ndim=2] prods
    cdef np.ndarray[np.int_t, ndim=2] freq
    cdef np.ndarray[np.double_t, ndim=2] sqi
    cdef np.ndarray[np.double_t, ndim=2] sqj
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sup = min_support

    prods = np.zeros((n, n), np.double)
    freq = np.zeros((n, n), np.int)
    sqi = np.zeros((n, n), np.double)
    sqj = np.zeros((n, n), np.double)
    sim = np.zeros((n, n), np.double)

    for k, v in data.items():
        for xi, ri in v:
            for xj, rj in v:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n):
            if freq[xi, xj] < min_sup:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum
                
            sim[xj, xi] = sim[xi, xj]

    return sim


def cosine(n_x, yr, min_support):
    cdef np.ndarray[np.double_t, ndim=2] prods
    cdef np.ndarray[np.int_t, ndim=2] freq
    cdef np.ndarray[np.double_t, ndim=2] sqi
    cdef np.ndarray[np.double_t, ndim=2] sqj
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim