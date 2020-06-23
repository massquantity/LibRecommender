import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport exp as cexp
from libcpp cimport bool
from libcpp.algorithm cimport binary_search
from libcpp.vector cimport vector

from ..utils.timing import time_block
from ..utils.misc import shuffle_data


cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937(unsigned int)
    
    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef bool check_consumed(const int[:] indices, const int[:] indptr, 
                         int user, int item_neg) nogil:
    return binary_search(&indices[indptr[user]], 
                         &indices[indptr[user+1]], 
                         item_neg)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef bpr_update(train_data, user_embed, item_embed, lr, reg, 
                 n_users, n_items, shuffle, num_threads, seed):

    if train_data.has_sampled:
        user_indices = train_data.user_indices_orig.astype(np.int32)
        item_indices = train_data.item_indices_orig.astype(np.int32)
    else:
        user_indices = train_data.user_indices.astype(np.int32)
        item_indices = train_data.item_indices.astype(np.int32)
    
    sparse_interaction = train_data.sparse_interaction
    sparse_indices = sparse_interaction.indices
    sparse_indptr = sparse_interaction.indptr

    if not reg:
        reg = 0.0

    if shuffle:
        user_indices, item_indices = shuffle_data(
            len(user_indices), user_indices, item_indices)

    _bpr_update(user_indices,
        item_indices,
        sparse_indices,
        sparse_indptr,
        user_embed,
        item_embed,
        lr,
        reg,
        n_users,
        n_items,
        num_threads,
        seed)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _bpr_update(const int[:] user_indices, const int[:] item_indices, 
                      const int[:] sparse_indices, const int[:] sparse_indptr, 
                      float[:, ::1] user_embed, float[:, ::1] item_embed, 
                      double lr, double reg, int n_users, int n_items, 
                      int num_threads, int seed):

    cdef Py_ssize_t i, j, t, random_seed, user, item_pos, item_neg
    cdef int length = len(user_indices), embed_size = user_embed.shape[1] - 1
    cdef float item_diff, log_sigmoid_grad, temp
    cdef long lower_bound = 0, upper_bound = n_items - 1

    cdef float *user_embed_ptr
    cdef float *item_pos_embed_ptr
    cdef float *item_neg_embed_ptr
    
    cdef vector[mt19937] rng
    cdef vector[uniform_int_distribution[long]] dist

    for i in range(num_threads):
        random_seed = (seed + i * 11) % 7
        rng.push_back(mt19937(random_seed))
        dist.push_back(uniform_int_distribution[long](
            lower_bound, upper_bound))

    with nogil, parallel(num_threads=num_threads):
        for i in prange(length):
            t = i % num_threads
            user = user_indices[i]
            item_pos = item_indices[i]
            item_neg = dist[t](rng[t])
            while check_consumed(sparse_indices, sparse_indptr, 
                                 user, item_neg):
                item_neg = dist[t](rng[t])

            user_embed_ptr = &user_embed[user, 0]
            item_pos_embed_ptr = &item_embed[item_pos, 0]
            item_neg_embed_ptr = &item_embed[item_neg, 0]

            item_diff = 0
            for j in range(embed_size + 1):
                item_diff = item_diff + user_embed_ptr[j] * (
                    item_pos_embed_ptr[j] - item_neg_embed_ptr[j])
            log_sigmoid_grad = 1.0 / (1.0 + cexp(item_diff))

            for j in range(embed_size):
                temp = user_embed_ptr[j]
                user_embed_ptr[j] += lr * (
                    log_sigmoid_grad * (
                        item_pos_embed_ptr[j] - item_neg_embed_ptr[j]
                    ) - reg * user_embed_ptr[j]
                )
                item_pos_embed_ptr[j] += lr * (
                    log_sigmoid_grad * temp - reg * item_pos_embed_ptr[j])
                item_neg_embed_ptr[j] += lr * (
                    -log_sigmoid_grad * temp - reg * item_neg_embed_ptr[j])

            item_pos_embed_ptr[embed_size] += lr * (
                log_sigmoid_grad - reg * item_pos_embed_ptr[embed_size])
            item_neg_embed_ptr[embed_size] += lr * (
                -log_sigmoid_grad - reg * item_neg_embed_ptr[embed_size])


