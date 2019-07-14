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



cimport cython
from libc.math cimport sqrt, pow
@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_cym(n, data, min_support=5):
    cdef double[:, :] prods = np.zeros((n, n), np.double)
    cdef int[:, :] freq = np.zeros((n, n), np.intc)
    cdef double[:, :] sqi = np.zeros((n, n), np.double)
    cdef double[:, :] sqj = np.zeros((n, n), np.double)
    cdef double[:, :] sim = np.zeros((n, n), np.double)

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sup = min_support

    for k, v in data.items():
        for xi, ri in v:
            for xj, rj in v:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += pow(ri, 2)
                sqj[xi, xj] += pow(rj, 2)

    for xi in range(n):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n):
            if freq[xi, xj] < min_sup:
                sim[xi, xj] = 0
            else:
                denum = sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum
                
            sim[xj, xi] = sim[xi, xj]

    return sim


@cython.boundscheck(False)
@cython.wraparound(False)
def sk_num(n, data):
    cdef np.ndarray[np.int_t, ndim=2] num = np.zeros((n, n), np.int)
    cdef int ui, uj

    for i, users in data.items():
        for ui in users:
            for uj in users:
                num[ui, uj] += 1
    return num