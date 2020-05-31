import time
import math
import logging
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, issparse
import itertools
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
try:
    from ._similarities import forward_cosine, invert_cosine
except ImportError:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warn("Cython version is not available")
    pass


def cosine_sim(sparse_data, num_x, num_y, block_size=None, num_threads=1, min_support=1, mode="invert"):
    if not block_size:
        block_size_ = 1024 * math.ceil(2e8 / num_x / 1024)
        print("block size: ", block_size_)

    indices = sparse_data.indices.astype(np.int32)
    indptr = sparse_data.indptr.astype(np.int32)
    data = sparse_data.data.astype(np.float32)
    n_x, n_y = num_x, num_y

    t0 = time.time()
    if mode == "forward":
        res_indices, res_indptr, res_data = forward_cosine(indices, indptr, data, min_support, n_x)
    elif mode == "invert":
        res_indices, res_indptr, res_data = invert_cosine(indices, indptr, data, min_support, n_x,
                                                          n_y, block_size_, num_threads)
    else:
        raise ValueError("mode must either be 'forward' or 'invert'")
    print(f"sim time: {(time.time() - t0):5.2f}")

    sim_upper_triangular = csr_matrix(
        (res_data, res_indices, res_indptr), shape=(n_x, n_x), dtype=np.float32)
    return sim_upper_triangular + sim_upper_triangular.transpose()

