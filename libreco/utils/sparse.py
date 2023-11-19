from dataclasses import dataclass
from typing import List

from scipy.sparse import csr_matrix


@dataclass
class SparseMatrix:
    sparse_indices: List[int]
    sparse_indptr: List[int]
    sparse_data: List[float]


def build_sparse(matrix: csr_matrix, transpose: bool = False):
    m = matrix.T.tocsr() if transpose else matrix
    return SparseMatrix(
        m.indices.tolist(),
        m.indptr.tolist(),
        m.data.tolist(),
    )
