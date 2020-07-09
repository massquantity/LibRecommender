#cython: language_level=3
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt as csqrt, pow as cpow
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

ctypedef unsigned int uint


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Py_ssize_t count_freq(
    const int[:] indices, 
    const int[:] indptr, 
    int min_common, 
    int n_x, 
    int n_y, 
    int block_size, 
    int block_num, 
    int n_threads
) nogil:

    cdef: 
        Py_ssize_t x_start, x_end, p, x1, x2, i, j, index
        Py_ssize_t block_index, block_start, block_end, scount
        Py_ssize_t res_count = 0
    cdef uint *pre_freq
    cdef uint *sum_array = <uint *> calloc(block_num, sizeof(uint))

    for block_index in prange(block_num, num_threads=n_threads):
        pre_freq = <uint *> malloc(sizeof(uint) * n_x * block_size)
        memset(pre_freq, 0, sizeof(uint) * n_x * block_size)
        block_start = block_index * block_size
        block_end = (
            n_x if n_x < block_start + block_size 
                else block_start + block_size
        )

        for p in range(n_y):
            x_start = indptr[p]
            x_end = indptr[p + 1]
            for i in range(x_start, x_end):
                x1 = indices[i]
                if x1 >= block_start and x1 < block_end:
                    for j in range(i + 1, x_end):
                        x2 = indices[j]
                        index = (x1 - block_start) * n_x + x2
                        pre_freq[index] += 1

        for x1 in range(block_start, block_end):
            for x2 in range(x1 + 1, n_x):
                index = (x1 - block_start) * n_x + x2
                scount = pre_freq[index]
                if scount >= min_common:
                    sum_array[block_index] += 1

        free(pre_freq)

    for i in range(block_num):
        res_count += sum_array[i]
    free(sum_array)
    return res_count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_cosine(
    const int[:] indices, 
    const int[:] indptr, 
    const float[:] data, 
    const float[:] x_norm, 
    int min_common, 
    int n_x, 
    int n_y, 
    int block_size, 
    int block_num, 
    int n_threads, 
    float[:] res_data, 
    uint[:] res_indices, 
    uint[:] res_indptr
) nogil:

    cdef: 
        Py_ssize_t x_start, x_end, p, x1, x2, i, j, index
        Py_ssize_t block_index, block_start, block_end, scount
        Py_ssize_t res_index = 0
    cdef float sprods, sqi, sqj, cosine

    cdef float *prods
    cdef uint *freq

    for block_index in range(block_num):
        prods = <float *> malloc(sizeof(float) * n_x * block_size)
        freq = <uint *> malloc(sizeof(uint) * n_x * block_size)
        memset(prods, 0, sizeof(float) * n_x * block_size)
        memset(freq, 0, sizeof(uint) * n_x * block_size)

        block_start = block_index * block_size
        block_end = (
            n_x if n_x < block_start + block_size 
                else block_start + block_size
        )

        for p in range(n_y):
            x_start = indptr[p]
            x_end = indptr[p + 1]
            for i in range(x_start, x_end):
                x1 = indices[i]
                if x1 >= block_start and x1 < block_end:
                    for j in range(i + 1, x_end):
                        x2 = indices[j]
                        index = (x1 - block_start) * n_x + x2
                        prods[index] += data[i] * data[j]
                        freq[index] += 1

        for x1 in range(block_start, block_end):
            for x2 in range(x1 + 1, n_x):
                index = (x1 - block_start) * n_x + x2
                scount = freq[index]
                if scount >= min_common:
                    sprods = prods[index]
                    sqi = x_norm[x1]
                    sqj = x_norm[x2]
                    if sprods == 0.0 or sqi == 0.0 or sqj == 0.0:
                        cosine = 0.0
                    else:
                        cosine = sprods / (sqi * sqj)
                    res_data[res_index] = cosine
                    res_indices[res_index] = x2
                    res_index += 1
            res_indptr[x1 + 1] = res_index

        free(prods)
        free(freq)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef invert_cosine(
    const int[:] indices, 
    const int[:] indptr, 
    const float[:] data, 
    const float[:] x_norm, 
    int min_common, 
    int n_x, 
    int n_y, 
    int block_size, 
    int block_num, 
    int num_threads=1
):

    cdef Py_ssize_t res_count = count_freq(indices, indptr, min_common, 
                                           n_x, n_y, block_size, block_num, 
                                           num_threads)

    cdef float[:] res_data = np.zeros(res_count, dtype=np.single)
    cdef uint[:] res_indices = np.zeros(res_count, dtype=np.uintc)
    cdef uint[:] res_indptr = np.zeros(n_x + 1, dtype=np.uintc)
#    res_indptr[0] = 0

    compute_cosine(indices, indptr, data, x_norm, min_common, n_x, n_y, 
        block_size, block_num, num_threads, res_data, res_indices, res_indptr)

    return np.asarray(res_indices), np.asarray(res_indptr), np.asarray(res_data)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_pearson(
    const int[:] indices, 
    const int[:] indptr, 
    const float[:] data, 
    const float[:] x_mean, 
    const float[:] x_mean_centered_norm, 
    int min_common, 
    int n_x, 
    int n_y, 
    int block_size, 
    int block_num, 
    int n_threads, 
    float[:] res_data, 
    uint[:] res_indices, 
    uint[:] res_indptr
) nogil:

    cdef: 
        Py_ssize_t x_start, x_end, p, x1, x2, i, j, index
        Py_ssize_t block_index, block_start, block_end, scount
        Py_ssize_t res_index = 0
    cdef float sprods, smeani, smeanj, sqi, sqj, pearson

    cdef float *prods
    cdef uint *freq

    for block_index in range(block_num):
        prods = <float *> malloc(sizeof(float) * n_x * block_size)
        freq = <uint *> malloc(sizeof(uint) * n_x * block_size)
        memset(prods, 0, sizeof(float) * n_x * block_size)
        memset(freq, 0, sizeof(uint) * n_x * block_size)

        block_start = block_index * block_size
        block_end = (
            n_x if n_x < block_start + block_size 
                else block_start + block_size
        )

        for p in range(n_y):
            x_start = indptr[p]
            x_end = indptr[p + 1]
            for i in range(x_start, x_end):
                x1 = indices[i]
                if x1 >= block_start and x1 < block_end:
                    for j in range(i + 1, x_end):
                        x2 = indices[j]
                        index = (x1 - block_start) * n_x + x2
                        smeani = data[i] - x_mean[x1]
                        smeanj = data[j] - x_mean[x2]
                        prods[index] += (smeani * smeanj)
                        freq[index] += 1

        for x1 in range(block_start, block_end):
            for x2 in range(x1 + 1, n_x):
                index = (x1 - block_start) * n_x + x2
                scount = freq[index]
                if scount >= min_common:
                    sprods = prods[index]
                    sqi = x_mean_centered_norm[x1]
                    sqj = x_mean_centered_norm[x2]
                    if sprods == 0.0 or sqi == 0.0 or sqj == 0.0:
                        pearson = 0.0
                    else:
                        pearson = sprods / (sqi * sqj)
                    res_data[res_index] = pearson
                    res_indices[res_index] = x2
                    res_index += 1
            res_indptr[x1 + 1] = res_index

        free(prods)
        free(freq)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef invert_pearson(
    const int[:] indices, 
    const int[:] indptr, 
    const float[:] data, 
    const float[:] x_mean, 
    const float[:] x_mean_centered_norm, 
    int min_common, 
    int n_x, 
    int n_y, 
    int block_size, 
    int block_num, 
    int num_threads=1
):

    cdef Py_ssize_t res_count = count_freq(
        indices, indptr, min_common, n_x, n_y, block_size, block_num, num_threads)

    cdef float[:] res_data = np.zeros(res_count, dtype=np.single)
    cdef uint[:] res_indices = np.zeros(res_count, dtype=np.uintc)
    cdef uint[:] res_indptr = np.zeros(n_x + 1, dtype=np.uintc)
#    res_indptr[0] = 0

    compute_pearson(indices, indptr, data, x_mean, x_mean_centered_norm, min_common, 
        n_x, n_y, block_size, block_num, num_threads, res_data, res_indices, res_indptr)

    return np.asarray(res_indices), np.asarray(res_indptr), np.asarray(res_data)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_jaccard(
    const int[:] indices, 
    const int[:] indptr, 
    const float[:] data, 
    const int[:] x_count, 
    int min_common, 
    int n_x, 
    int n_y, 
    int block_size, 
    int block_num, 
    int n_threads, 
    float[:] res_data, 
    uint[:] res_indices, 
    uint[:] res_indptr
) nogil:

    cdef: 
        Py_ssize_t x_start, x_end, p, x1, x2, i, j, index
        Py_ssize_t block_index, block_start, block_end, scount
        Py_ssize_t res_index = 0
    cdef float jaccard, intersection, union
    cdef uint *freq

    for block_index in range(block_num):
        freq = <uint *> malloc(sizeof(uint) * n_x * block_size)
        memset(freq, 0, sizeof(uint) * n_x * block_size)

        block_start = block_index * block_size
        block_end = (
            n_x if n_x < block_start + block_size 
                else block_start + block_size
        )

        for p in range(n_y):
            x_start = indptr[p]
            x_end = indptr[p + 1]
            for i in range(x_start, x_end):
                x1 = indices[i]
                if x1 >= block_start and x1 < block_end:
                    for j in range(i + 1, x_end):
                        x2 = indices[j]
                        index = (x1 - block_start) * n_x + x2
                        freq[index] += 1

        for x1 in range(block_start, block_end):
            for x2 in range(x1 + 1, n_x):
                index = (x1 - block_start) * n_x + x2
                intersection = freq[index]
                if intersection >= min_common:
                    union = x_count[x1] + x_count[x2] - intersection
                    jaccard = intersection / union
                    res_data[res_index] = jaccard
                    res_indices[res_index] = x2
                    res_index += 1
            res_indptr[x1 + 1] = res_index

        free(freq)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef invert_jaccard(const int[:] indices, const int[:] indptr, const float[:] data, 
                    const int[:] x_count, int min_common, int n_x, int n_y, 
                    int block_size, int block_num, int num_threads=1):

    cdef Py_ssize_t res_count = count_freq(
        indices, indptr, min_common, n_x, n_y, block_size, block_num, num_threads)

    cdef float[:] res_data = np.zeros(res_count, dtype=np.single)
    cdef uint[:] res_indices = np.zeros(res_count, dtype=np.uintc)
    cdef uint[:] res_indptr = np.zeros(n_x + 1, dtype=np.uintc)
#    res_indptr[0] = 0

    compute_jaccard(indices, indptr, data, x_count, min_common, n_x, n_y, 
        block_size, block_num, num_threads, res_data, res_indices, res_indptr)

    return np.asarray(res_indices), np.asarray(res_indptr), np.asarray(res_data)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef forward_cosine(const int[:] indices, const int[:] indptr, 
                     const float[:] data, const float[:] x_norm, 
                     int min_common, int n_x):

    cdef Py_ssize_t x1, x2, y1, y2, i, j, end1, end2, count
    cdef float cos, prods, sqi, sqj
    cdef vector[uint] res_indices, res_indptr
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
                        i += 1
                        j += 1

                if count >= min_common:
                    res_indices.push_back(x2)
                    sqi = x_norm[x1]
                    sqj = x_norm[x2]
                    if prods == 0.0 or sqi == 0.0 or sqj == 0.0:
                        cos = 0.0
                    else:
                        cos = prods / (sqi * sqj)
                    res_data.push_back(cos)
            res_indptr.push_back(res_indices.size())

    return res_indices, res_indptr, res_data


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef forward_pearson(const int[:] indices, const int[:] indptr, 
                      const float[:] data, const float[:] x_mean, 
                      const float[:] x_mean_centered_norm, 
                      int min_common, int n_x):

    cdef Py_ssize_t x1, x2, y1, y2, i, j, end1, end2, count
    cdef float pearson, prods, sqi, sqj, smean1, smean2
    cdef vector[uint] res_indices, res_indptr
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
                        smean1 = data[i] - x_mean[x1]
                        smean2 = data[j] - x_mean[x2]
                        prods += (smean1 * smean2)
                        i += 1
                        j += 1

                if count >= min_common:
                    res_indices.push_back(x2)
                    sqi = x_mean_centered_norm[x1]
                    sqj = x_mean_centered_norm[x2]
                    if prods == 0.0 or sqi == 0.0 or sqj == 0.0:
                        pearson = 0.0
                    else:
                        pearson = prods / (sqi * sqj)
                    res_data.push_back(pearson)
            res_indptr.push_back(res_indices.size())

    return res_indices, res_indptr, res_data


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef forward_jaccard(const int[:] indices, const int[:] indptr, 
                      const float[:] data, const int[:] x_count, 
                      int min_common, int n_x):

    cdef Py_ssize_t x1, x2, y1, y2, i, j, end1, end2
    cdef float jaccard, intersection, union
    cdef vector[uint] res_indices, res_indptr
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

                intersection = 0.0
                while (i < end1 and j < end2):
                    y1 = indices[i]
                    y2 = indices[j]
                    if y1 < y2:
                        i += 1
                    elif y1 > y2:
                        j += 1
                    else:
                        intersection += 1
                        i += 1
                        j += 1

                if intersection >= min_common:
                    res_indices.push_back(x2)
                    union = x_count[x1] + x_count[x2] - intersection
                    jaccard = intersection / union
                    res_data.push_back(jaccard)
            res_indptr.push_back(res_indices.size())

    return res_indices, res_indptr, res_data

