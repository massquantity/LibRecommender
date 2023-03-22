import numpy as np


def construct_unique_feat(
    user_indices,
    item_indices,
    sparse_indices,
    dense_values,
    col_name_mapping,
    unique_feat,
):
    # use mergesort to preserve order
    sort_kind = "quicksort" if unique_feat else "mergesort"
    user_pos = np.argsort(user_indices, kind=sort_kind)
    item_pos = np.argsort(item_indices, kind=sort_kind)

    user_sparse_matrix, item_sparse_matrix = None, None
    user_dense_matrix, item_dense_matrix = None, None
    if "user_sparse_col" in col_name_mapping:
        user_sparse_col = list(col_name_mapping["user_sparse_col"].values())
        user_sparse_matrix = _compress_unique_values(
            sparse_indices, user_sparse_col, user_indices, user_pos
        )
    if "item_sparse_col" in col_name_mapping:
        item_sparse_col = list(col_name_mapping["item_sparse_col"].values())
        item_sparse_matrix = _compress_unique_values(
            sparse_indices, item_sparse_col, item_indices, item_pos
        )
    if "user_dense_col" in col_name_mapping:
        user_dense_col = list(col_name_mapping["user_dense_col"].values())
        user_dense_matrix = _compress_unique_values(
            dense_values, user_dense_col, user_indices, user_pos
        )
    if "item_dense_col" in col_name_mapping:
        item_dense_col = list(col_name_mapping["item_dense_col"].values())
        item_dense_matrix = _compress_unique_values(
            dense_values, item_dense_col, item_indices, item_pos
        )
    return (
        user_sparse_matrix,
        user_dense_matrix,
        item_sparse_matrix,
        item_dense_matrix,
    )


# https://stackoverflow.com/questions/46390376/drop-duplicates-from-structured-numpy-array-python3-x
def _compress_unique_values(orig_val, col, indices, pos):
    values = np.take(orig_val, col, axis=1)
    values = values.reshape(-1, 1) if orig_val.ndim == 1 else values
    indices = indices[pos]
    mask = np.empty(len(indices), dtype=bool)
    mask[:-1] = indices[:-1] != indices[1:]
    mask[-1] = True
    mask = pos[mask]
    unique_values = values[mask]
    assert len(np.unique(indices)) == len(unique_values)
    return unique_values
