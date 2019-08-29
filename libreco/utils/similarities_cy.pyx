cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt, pow


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_cy(n, data, min_support=5):
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
        sim[xi, xi] = 1.0
        for xj in range(xi + 1, n):
            if freq[xi, xj] < min_sup:
                sim[xi, xj] = 0.0
            else:
                denum = sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum
                
            sim[xj, xi] = sim[xi, xj]

    return sim



@cython.boundscheck(False)
@cython.wraparound(False)
def pearson_cy(n, data, min_support=5):
    cdef double[:, :] prods = np.zeros((n, n), np.double)
    cdef int[:, :] freq = np.zeros((n, n), np.intc)
    cdef double[:, :] sqi = np.zeros((n, n), np.double)
    cdef double[:, :] sqj = np.zeros((n, n), np.double)
    cdef double[:, :] si = np.zeros((n, n), np.double)
    cdef double[:, :] sj = np.zeros((n, n), np.double)
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
                si[xi, xj] += ri
                sj[xi, xj] += rj

    for xi in range(n):
        sim[xi, xi] = 1.0
        for xj in range(xi + 1, n):
            if freq[xi, xj] < min_sup:
                sim[xi, xj] = 0.0
            else:
                num = freq[xi, xj] * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = sqrt(
                    (freq[xi, xj] * sqi[xi, xj] - si[xi, xj] * si[xi, xj]) * 
                    (freq[xi, xj] * sqj[xi, xj] - sj[xi, xj] * sj[xi, xj]))

                if denum == 0.0:
                    sim[xi, xj] = 0.0
                else:
                    sim[xi, xj] = num / denum
                
            sim[xj, xi] = sim[xi, xj]

    return sim

@cython.boundscheck(False)
@cython.wraparound(False)
def sk_num(n, data):
    cdef np.ndarray[np.int_t, ndim=2] num = np.zeros((n, n), np.intc)
    cdef int ui, uj

    for i, users in data.items():
        for ui in users:
            for uj in users:
                num[ui, uj] += 1
    return num