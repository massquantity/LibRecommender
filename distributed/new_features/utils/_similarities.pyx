from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt as csqrt, pow as cpow
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef forward_cosine(const int[:] indices, const int[:] indptr, const float[:] data,
                      int min_support, int n_x):
    cdef Py_ssize_t x1, x2, y1, y2, i, j, end1, end2, count
    cdef float cos, prods, sqi, sqj
    cdef vector[unsigned int] res_indices, res_indptr
    cdef vector[float] res_data
    res_indptr.reserve(n_x + 1)
    res_indptr.push_back(0)

    with nogil:
        for x1 in range(n_x):
            for x2 in range(x1 + 1, n_x):
                i = indptr[x1]
                j = indptr[x2]
                end1 = indptr[x1 + 1]
                end2 = indptr[x2 + 1]

                prods = 0.0
                sqi = 0.0
                sqj = 0.0
                count = 0
                # compute common items
                while (i < end1 and j < end2):
                    y1 = indices[i]
                    y2 = indices[j]
                    if y1 < y2:
                        i += 1
                    elif y1 > y2:
                        j += 1
                    else:
                        count += 1
                        prods += data[i] * data[j]
                        sqi += cpow(data[i], 2)
                        sqj += cpow(data[j], 2)
                        i += 1
                        j += 1

                if count >= min_support:
                    res_indices.push_back(x2)
                    cos = prods / csqrt(sqi * sqj)
                    res_data.push_back(cos)
            res_indptr.push_back(res_indices.size())

    return res_indices, res_indptr, res_data


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef invert_cosine(const int[:] indices, const int[:] indptr, const float[:] data,
                    int min_support, int n_x, int n_y, int block_size, int num_threads=1):

    cdef np.ndarray[np.float32_t, ndim=1] res_data
    cdef np.ndarray[np.uint32_t, ndim=1] res_indices
    cdef np.ndarray[np.uint32_t, ndim=1] res_indptr

    cdef Py_ssize_t begin, end, p, x1, x2, i, j, index, n, scount, res_count = 0, res_index = 0
    cdef float sprods, ssqi, ssqj, cosine
    cdef int n_threads = num_threads
    cdef Py_ssize_t last

    cdef unsigned int *pre_freq
    cdef float *prods
    cdef unsigned int *freq
    cdef float *sqi
    cdef float *sqj

    with nogil:
        for n from 0 <= n < n_x by block_size:
            try:
                pre_freq = <unsigned int *> malloc(sizeof(unsigned int) * n_x * block_size)
                memset(pre_freq, 0, sizeof(unsigned int) * n_x * block_size)
                for p in prange(n_y, num_threads=n_threads):
                    begin = indptr[p]
                    end = indptr[p + 1]
                    for x1 in range(begin, end):
                        i = indices[x1]
                        if i >= n and i < n + block_size:
                            for x2 in range(x1 + 1, end):
                                j = indices[x2]
                                index = (i - n) * n_x + j
                                pre_freq[index] += 1

                last = n_x if n_x < n + block_size else n + block_size
                for x1 in range(n, last):
                    for x2 in range(x1 + 1, n_x):
                        index = (x1 - n) * n_x + x2
                        scount = pre_freq[index]
                        if scount >= min_support:
                            res_count += 1

            finally:
                free(pre_freq)

        with gil:
            res_data = np.zeros(res_count, dtype=np.single)
            res_indices = np.zeros(res_count, dtype=np.uintc)
            res_indptr = np.zeros(n_x + 1, dtype=np.uintc)
        #    res_indptr[0] = 0

        for n from 0 <= n < n_x by block_size:
            try:
                prods = <float *> malloc(sizeof(float) * n_x * block_size)
                freq = <unsigned int *> malloc(sizeof(unsigned int) * n_x * block_size)
                sqi = <float *> malloc(sizeof(float) * n_x * block_size)
                sqj = <float *> malloc(sizeof(float) * n_x * block_size)
                memset(prods, 0, sizeof(float) * n_x * block_size)
                memset(freq, 0, sizeof(unsigned int) * n_x * block_size)
                memset(sqi, 0, sizeof(float) * n_x * block_size)
                memset(sqj, 0, sizeof(float) * n_x * block_size)

                for p in prange(n_y, num_threads=n_threads):
                    begin = indptr[p]
                    end = indptr[p + 1]
                    for x1 in range(begin, end):
                        i = indices[x1]
                        if i >= n and i < n + block_size:
                            for x2 in range(x1 + 1, end):
                                j = indices[x2]
                                index = (i - n) * n_x + j
                                prods[index] += data[x1] * data[x2]
                                sqi[index] += cpow(data[x1], 2)
                                sqj[index] += cpow(data[x2], 2)
                                freq[index] += 1

                last = n_x if n_x < n + block_size else n + block_size
                for x1 in range(n, last):
                    for x2 in range(x1 + 1, n_x):
                        index = (x1 - n) * n_x + x2
                        scount = freq[index]
                        if scount >= min_support:
                            sprods = prods[index]
                            ssqi = sqi[index]
                            ssqj = sqj[index]
                            cosine = sprods / csqrt(ssqi * ssqj)
                            res_data[res_index] = cosine
                            res_indices[res_index] = x2
                            res_index += 1
                    res_indptr[x1 + 1] = res_index

            finally:
                free(prods)
                free(freq)
                free(sqi)
                free(sqj)

    return res_indices, res_indptr, res_data