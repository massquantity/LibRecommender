from collections import defaultdict, OrderedDict
import numpy as np


def col_name2index(user_col=None, item_col=None,
                   sparse_col=None, dense_col=None):
    # format: {column_family_name: {column_name: index}}
    # if no such family, default format would be: {column_family_name: {[]: []}
    name_mapping = defaultdict(OrderedDict)
    if sparse_col:
        sparse_col_dict = {col: i for i, col in enumerate(sparse_col)}
        name_mapping["sparse_col"].update(sparse_col_dict)
    if dense_col:
        dense_col_dict = {col: i for i, col in enumerate(dense_col)}
        name_mapping["dense_col"].update(dense_col_dict)

    if user_col and sparse_col:
        user_sparse_col = _extract_common_col(sparse_col, user_col)
        for col in user_sparse_col:
            name_mapping["user_sparse_col"].update(
                {col: name_mapping["sparse_col"][col]}
            )
    if user_col and dense_col:
        user_dense_col = _extract_common_col(dense_col, user_col)
        for col in user_dense_col:
            name_mapping["user_dense_col"].update(
                {col: name_mapping["dense_col"][col]}
            )

    if item_col and sparse_col:
        item_sparse_col = _extract_common_col(sparse_col, item_col)
        for col in item_sparse_col:
            name_mapping["item_sparse_col"].update(
                {col: name_mapping["sparse_col"][col]}
            )
    if item_col and dense_col:
        item_dense_col = _extract_common_col(dense_col, item_col)
        for col in item_dense_col:
            name_mapping["item_dense_col"].update(
                {col: name_mapping["dense_col"][col]}
            )

    return name_mapping


def _extract_common_col(col1, col2):
    # np.intersect1d will return the sorted common column names,
    # but we also want to preserve the original order of common column in
    # col1 and col2
    common_col, indices_in_col1, _ = np.intersect1d(col1, col2,
                                                    assume_unique=True,
                                                    return_indices=True)
    return common_col[np.lexsort((common_col, indices_in_col1))]

