import time
import math
import logging
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as spnorm
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
try:
    from ._similarities import (
        forward_cosine,
        invert_cosine,
        forward_pearson,
        invert_pearson,
        forward_jaccard,
        invert_jaccard
    )
except (ImportError, ModuleNotFoundError):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warning("Similarity cython version is not available")
    pass


def _choose_blocks(num, b_size=None):
    # To calculate an n by n similarity matrix (n is num of user or item),
    # memory is usually a big concern, so the matrix is divided into several
    # blocks to calculate separately. The default block size is 1024, so num
    # of elements in a block is 2e8, which is roughly 1024 * 200000 users/items.
    # Of course num of users/items will vary in various datasets,
    # so here default block num is defined as how many blocks can exist,
    # and block size is calculated to make sure data are divided evenly.
    if not b_size:
        block_num = math.ceil(num / (2e8 / num))
        block_size = math.ceil(num / block_num)
    else:
        block_size = b_size
        block_num = math.ceil(num / block_size)
    print(f"Final block size and num: {block_size, block_num}")
    return block_size, block_num


def cosine_sim(sparse_data_x, sparse_data_y, num_x, num_y, block_size=None,
               num_threads=1, min_common=1, mode="invert"):
    block_size, block_num = _choose_blocks(num_x, block_size)
    n_x, n_y = num_x, num_y

    if mode == "forward":
        indices = sparse_data_x.indices.astype(np.int32)
        indptr = sparse_data_x.indptr.astype(np.int32)
        data = sparse_data_x.data.astype(np.float32)
        x_norm = compute_sparse_norm(sparse_data_x)

        res_indices, res_indptr, res_data = forward_cosine(
            indices, indptr, data, x_norm, min_common, n_x)

    elif mode == "invert":
        indices = sparse_data_y.indices.astype(np.int32)
        indptr = sparse_data_y.indptr.astype(np.int32)
        data = sparse_data_y.data.astype(np.float32)
        x_norm = compute_sparse_norm(sparse_data_x)

        res_indices, res_indptr, res_data = invert_cosine(
            indices, indptr, data, x_norm, min_common, n_x, n_y,
            block_size, block_num, num_threads)

    else:
        raise ValueError("mode must either be 'forward' or 'invert'")

    sim_upper_triangular = csr_matrix(
        (res_data, res_indices, res_indptr),
        shape=(n_x, n_x), dtype=np.float32
    )
    return sim_upper_triangular + sim_upper_triangular.transpose()


def pearson_sim(sparse_data_x, sparse_data_y, num_x, num_y, block_size=None,
                num_threads=1, min_common=1, mode="invert"):
    block_size, block_num = _choose_blocks(num_x, block_size)
    n_x, n_y = num_x, num_y

    if mode == "forward":
        indices = sparse_data_x.indices.astype(np.int32)
        indptr = sparse_data_x.indptr.astype(np.int32)
        data = sparse_data_x.data.astype(np.float32)
        x_mean = compute_sparse_mean(sparse_data_x)
        x_mean_centered_norm = compute_sparse_mean_centered_norm(sparse_data_x)

        res_indices, res_indptr, res_data = forward_pearson(
            indices, indptr, data, x_mean, x_mean_centered_norm,
            min_common, n_x)

    elif mode == "invert":
        indices = sparse_data_y.indices.astype(np.int32)
        indptr = sparse_data_y.indptr.astype(np.int32)
        data = sparse_data_y.data.astype(np.float32)
        x_mean = compute_sparse_mean(sparse_data_x)
        x_mean_centered_norm = compute_sparse_mean_centered_norm(sparse_data_x)

        res_indices, res_indptr, res_data = invert_pearson(
            indices, indptr, data, x_mean, x_mean_centered_norm, min_common,
            n_x, n_y, block_size, block_num, num_threads)

    else:
        raise ValueError("mode must either be 'forward' or 'invert'")

    sim_upper_triangular = csr_matrix(
        (res_data, res_indices, res_indptr),
        shape=(n_x, n_x), dtype=np.float32
    )
    return sim_upper_triangular + sim_upper_triangular.transpose()


def jaccard_sim(sparse_data_x, sparse_data_y, num_x, num_y, block_size=None,
                num_threads=1, min_common=1, mode="invert"):
    block_size, block_num = _choose_blocks(num_x, block_size)
    n_x, n_y = num_x, num_y

    if mode == "forward":
        indices = sparse_data_x.indices.astype(np.int32)
        indptr = sparse_data_x.indptr.astype(np.int32)
        data = sparse_data_x.data.astype(np.float32)
        x_count = compute_sparse_count(sparse_data_x)

        res_indices, res_indptr, res_data = forward_jaccard(
            indices, indptr, data, x_count, min_common, n_x)

    elif mode == "invert":
        indices = sparse_data_y.indices.astype(np.int32)
        indptr = sparse_data_y.indptr.astype(np.int32)
        data = sparse_data_y.data.astype(np.float32)
        x_count = compute_sparse_count(sparse_data_x)

        res_indices, res_indptr, res_data = invert_jaccard(
            indices, indptr, data, x_count, min_common,
            n_x, n_y, block_size, block_num, num_threads)

    else:
        raise ValueError("mode must either be 'forward' or 'invert'")

    sim_upper_triangular = csr_matrix(
        (res_data, res_indices, res_indptr),
        shape=(n_x, n_x), dtype=np.float32
    )
    return sim_upper_triangular + sim_upper_triangular.transpose()


def compute_sparse_norm(sparse_data):
    sparse_norm = spnorm(sparse_data, axis=1)
    return sparse_norm.astype(np.float32)


def compute_sparse_mean(sparse_data):
    # only consider interacted data
    x_sum = np.asarray(sparse_data.sum(axis=1)).flatten()
    x_num = np.diff(sparse_data.indptr)
    sparse_mean = x_sum / x_num
    return sparse_mean.astype(np.float32)


def compute_sparse_mean_centered_norm(sparse_data):
    # mainly for denominator of pearson correlation formula
    # only consider interacted data
    assert np.issubdtype(sparse_data.dtype, np.floating), (
        "sparse_data type must be float...")
    indices = sparse_data.indices.copy()
    indptr = sparse_data.indptr.copy()
    data = sparse_data.data.copy()
    length = sparse_data.shape[0]
    for x in range(length):
        x_slice = slice(indptr[x], indptr[x+1])
        x_mean = np.mean(data[x_slice])
        data[x_slice] -= x_mean
    sparse_data_mean_centered = csr_matrix((data, indices, indptr),
                                           shape=sparse_data.shape)
    return compute_sparse_norm(sparse_data_mean_centered)


def compute_sparse_count(sparse_data):
    return np.diff(sparse_data.indptr)

