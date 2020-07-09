import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport exp as cexp, pow as cpow, sqrt as csqrt
from libcpp cimport bool
from libcpp.algorithm cimport binary_search
from libcpp.vector cimport vector

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
cpdef bpr_update(optimizer, train_data, user_embed, item_embed, lr, reg, 
                 n_users, n_items, shuffle, num_threads, seed, epoch, 
                 u_velocity=None, i_velocity=None, momentum=0.9, 
                 u_1st_mom=None, i_1st_mom=None, u_2nd_mom=None, 
                 i_2nd_mom=None, rho1=0.9, rho2=0.999):

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

    if optimizer == "sgd":
        _bpr_update_sgd(user_indices,
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

    elif optimizer == "momentum":
        _bpr_update_momentum(user_indices,
                             item_indices,
                             sparse_indices,
                             sparse_indptr,
                             user_embed,
                             item_embed,
                             lr,
                             reg,
                             n_users,
                             n_items,
                             u_velocity, 
                             i_velocity,  
                             momentum, 
                             num_threads,
                             seed)

    elif optimizer == "adam":
        _bpr_update_adam(user_indices,
                         item_indices,
                         sparse_indices,
                         sparse_indptr,
                         user_embed,
                         item_embed,
                         lr,
                         reg,
                         n_users,
                         n_items,
                         u_1st_mom, 
                         i_1st_mom, 
                         u_2nd_mom, 
                         i_2nd_mom, 
                         rho1, 
                         rho2, 
                         epoch, 
                         num_threads,
                         seed)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _bpr_update_sgd(const int[:] user_indices, 
                          const int[:] item_indices, 
                          const int[:] sparse_indices, 
                          const int[:] sparse_indptr, 
                          float[:, ::1] user_embed, 
                          float[:, ::1] item_embed, 
                          double lr, 
                          double reg, 
                          int n_users, 
                          int n_items, 
                          int num_threads, 
                          int seed):

    cdef Py_ssize_t i, j, t, random_seed, user, item_pos, item_neg
    cdef int length = len(user_indices), embed_size = user_embed.shape[1] - 1
    cdef float item_diff, log_sigmoid_grad
    cdef float user_grad, item_pos_grad, item_neg_grad
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
                user_grad = log_sigmoid_grad * (
                    item_pos_embed_ptr[j] - item_neg_embed_ptr[j]
                ) - reg * user_embed_ptr[j]
                item_pos_grad = (
                    log_sigmoid_grad * user_embed_ptr[j] 
                    - reg * item_pos_embed_ptr[j]
                )
                item_neg_grad = (
                    - log_sigmoid_grad * user_embed_ptr[j] 
                    - reg * item_neg_embed_ptr[j]
                )

                user_embed_ptr[j] += lr * user_grad
                item_pos_embed_ptr[j] += lr * item_pos_grad
                item_neg_embed_ptr[j] += lr * item_neg_grad

            item_pos_embed_ptr[embed_size] += lr * (
                log_sigmoid_grad - reg * item_pos_embed_ptr[embed_size])
            item_neg_embed_ptr[embed_size] += lr * (
                -log_sigmoid_grad - reg * item_neg_embed_ptr[embed_size])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _bpr_update_momentum(const int[:] user_indices, 
                               const int[:] item_indices, 
                               const int[:] sparse_indices, 
                               const int[:] sparse_indptr, 
                               float[:, ::1] user_embed, 
                               float[:, ::1] item_embed, 
                               double lr, 
                               double reg, 
                               int n_users, 
                               int n_items, 
                               float[:, ::1] u_velocity, 
                               float[:, ::1] i_velocity, 
                               double momentum, 
                               int num_threads, 
                               int seed):

    cdef Py_ssize_t i, j, t, random_seed, user, item_pos, item_neg
    cdef int length = len(user_indices), embed_size = user_embed.shape[1] - 1
    cdef float item_diff, log_sigmoid_grad
    cdef float user_grad, item_pos_grad, item_neg_grad
    cdef long lower_bound = 0, upper_bound = n_items - 1

    cdef float *user_embed_ptr
    cdef float *item_pos_embed_ptr
    cdef float *item_neg_embed_ptr

    cdef float *user_embed_v_ptr
    cdef float *item_pos_embed_v_ptr
    cdef float *item_neg_embed_v_ptr
    
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

            user_embed_v_ptr = &u_velocity[user, 0]
            item_pos_embed_v_ptr = &i_velocity[item_pos, 0]
            item_neg_embed_v_ptr = &i_velocity[item_neg, 0]

            item_diff = 0
            for j in range(embed_size + 1):
                item_diff = item_diff + user_embed_ptr[j] * (
                    item_pos_embed_ptr[j] - item_neg_embed_ptr[j])
            log_sigmoid_grad = 1.0 / (1.0 + cexp(item_diff))

            for j in range(embed_size + 1):
                if j < embed_size:
                    user_grad = log_sigmoid_grad * (
                        item_pos_embed_ptr[j] - item_neg_embed_ptr[j]
                    ) - reg * user_embed_ptr[j]
                item_pos_grad = (
                    log_sigmoid_grad * user_embed_ptr[j] 
                    - reg * item_pos_embed_ptr[j]
                )
                item_neg_grad = (
                    - log_sigmoid_grad * user_embed_ptr[j] 
                    - reg * item_neg_embed_ptr[j]
                )

                if j < embed_size:
                    user_embed_v_ptr[j] = (
                        momentum * user_embed_v_ptr[j] + lr * user_grad)
                    user_embed_ptr[j] += user_embed_v_ptr[j]

                item_pos_embed_v_ptr[j] = (
                    momentum * item_pos_embed_v_ptr[j] + lr * item_pos_grad)
                item_pos_embed_ptr[j] += item_pos_embed_v_ptr[j]

                item_neg_embed_v_ptr[j] = (
                    momentum * item_neg_embed_v_ptr[j] + lr * item_neg_grad)
                item_neg_embed_ptr[j] += item_neg_embed_v_ptr[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _bpr_update_adam(const int[:] user_indices, 
                           const int[:] item_indices, 
                           const int[:] sparse_indices, 
                           const int[:] sparse_indptr, 
                           float[:, ::1] user_embed, 
                           float[:, ::1] item_embed, 
                           double lr, 
                           double reg, 
                           int n_users, 
                           int n_items, 
                           float[:, ::1] u_1st_mom, 
                           float[:, ::1] i_1st_mom, 
                           float[:, ::1] u_2nd_mom, 
                           float[:, ::1] i_2nd_mom,  
                           double rho1, 
                           double rho2, 
                           int epoch, 
                           int num_threads, 
                           int seed):

    cdef Py_ssize_t i, j, t, random_seed, user, item_pos, item_neg
    cdef int length = len(user_indices), embed_size = user_embed.shape[1] - 1
    cdef float item_diff, log_sigmoid_grad
    cdef float user_grad, item_pos_grad, item_neg_grad
    cdef float unbias_v_user, unbias_h_user
    cdef float unbias_v_pos_item, unbias_h_pos_item
    cdef float unbias_v_neg_item, unbias_h_neg_item
    cdef long lower_bound = 0, upper_bound = n_items - 1

    cdef float *user_embed_ptr
    cdef float *item_pos_embed_ptr
    cdef float *item_neg_embed_ptr

    cdef float *user_embed_v_ptr
    cdef float *user_embed_h_ptr
    cdef float *item_pos_embed_v_ptr
    cdef float *item_pos_embed_h_ptr
    cdef float *item_neg_embed_v_ptr
    cdef float *item_neg_embed_h_ptr
    
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

            user_embed_v_ptr = &u_1st_mom[user, 0]
            user_embed_h_ptr = &u_2nd_mom[user, 0]
            item_pos_embed_v_ptr = &i_1st_mom[item_pos, 0]
            item_pos_embed_h_ptr = &i_2nd_mom[item_pos, 0]
            item_neg_embed_v_ptr = &i_1st_mom[item_neg, 0]
            item_neg_embed_h_ptr = &i_2nd_mom[item_neg, 0]

            item_diff = 0
            for j in range(embed_size + 1):
                item_diff = item_diff + user_embed_ptr[j] * (
                    item_pos_embed_ptr[j] - item_neg_embed_ptr[j])
            log_sigmoid_grad = 1.0 / (1.0 + cexp(item_diff))

            for j in range(embed_size + 1):
                if j < embed_size:
                    user_grad = log_sigmoid_grad * (
                        item_pos_embed_ptr[j] - item_neg_embed_ptr[j]
                    ) - reg * user_embed_ptr[j]
                item_pos_grad = (
                    log_sigmoid_grad * user_embed_ptr[j] 
                    - reg * item_pos_embed_ptr[j]
                )
                item_neg_grad = (
                    - log_sigmoid_grad * user_embed_ptr[j] 
                    - reg * item_neg_embed_ptr[j]
                )

                if j < embed_size:
                    user_embed_v_ptr[j] = rho1 * user_embed_v_ptr[j] + (
                        1.0 - rho1) * user_grad
                    user_embed_h_ptr[j] = rho2 * user_embed_h_ptr[j] + (
                        1.0 - rho2) * cpow(user_grad, 2)
                    unbias_v_user = user_embed_v_ptr[j] / (
                        1.0 - cpow(rho1, epoch))
                    unbias_h_user = user_embed_h_ptr[j] / (
                        1.0 - cpow(rho2, epoch))
                    user_embed_ptr[j] = user_embed_ptr[j] + (
                        lr * unbias_v_user / (csqrt(unbias_h_user) + 1e-8))

                item_pos_embed_v_ptr[j] = rho1 * item_pos_embed_v_ptr[j] + (
                    1.0 - rho1) * item_pos_grad
                item_pos_embed_h_ptr[j] = rho2 * item_pos_embed_h_ptr[j] + (
                    1.0 - rho2) * cpow(item_pos_grad, 2)
                unbias_v_pos_item = item_pos_embed_v_ptr[j] / (
                    1.0 - cpow(rho1, epoch))
                unbias_h_pos_item = item_pos_embed_h_ptr[j] / (
                    1.0 - cpow(rho2, epoch))
                item_pos_embed_ptr[j] = item_pos_embed_ptr[j] + (
                    lr * unbias_v_pos_item / (csqrt(unbias_h_pos_item) + 1e-8))

                item_neg_embed_v_ptr[j] = rho1 * item_neg_embed_v_ptr[j] + (
                    1.0 - rho1) * item_neg_grad
                item_neg_embed_h_ptr[j] = rho2 * item_neg_embed_h_ptr[j] + (
                    1.0 - rho2) * cpow(item_neg_grad, 2)
                unbias_v_neg_item = item_neg_embed_v_ptr[j] / (
                    1.0 - cpow(rho1, epoch))
                unbias_h_neg_item = item_neg_embed_h_ptr[j] / (
                    1.0 - cpow(rho2, epoch))
                item_neg_embed_ptr[j] = item_neg_embed_ptr[j] + (
                    lr * unbias_v_neg_item / (csqrt(unbias_h_neg_item) + 1e-8))

