import numbers
import numpy as np
from ..utils.misc import colorize


def construct_unique_feat(user_indices, item_indices, sparse_indices,
                          dense_values, user_sparse_col, user_dense_col,
                          item_sparse_col, item_dense_col, unique_feat):
    # use mergesort to preserve order
    sort_kind = "quicksort" if unique_feat else "mergesort"
    user_pos = np.argsort(user_indices, kind=sort_kind)
    item_pos = np.argsort(item_indices, kind=sort_kind)

    if user_sparse_col:
        user_sparse_matrix = _compress_unique_values(
            sparse_indices, user_sparse_col, user_indices, user_pos)
    else:
        user_sparse_matrix = None

    if item_sparse_col:
        item_sparse_matrix = _compress_unique_values(
            sparse_indices, item_sparse_col, item_indices, item_pos)
    else:
        item_sparse_matrix = None

    if user_dense_col:
        user_dense_matrix = _compress_unique_values(
            dense_values, user_dense_col, user_indices, user_pos)
    else:
        user_dense_matrix = None

    if item_dense_col:
        item_dense_matrix = _compress_unique_values(
            dense_values, item_dense_col, item_indices, item_pos)
    else:
        item_dense_matrix = None

    rets = (user_sparse_matrix, user_dense_matrix,
            item_sparse_matrix, item_dense_matrix)
    return rets


def _compress_unique_values(orig_val, col, indices, pos):
    values = np.take(orig_val, col, axis=1)
    values = values.reshape(-1, 1) if orig_val.ndim == 1 else values
    indices = indices[pos]
    # https://stackoverflow.com/questions/46390376/drop-duplicates-from-structured-numpy-array-python3-x
    mask = np.empty(len(indices), dtype=np.bool)
    mask[:-1] = (indices[:-1] != indices[1:])
    mask[-1] = True
    mask = pos[mask]
    unique_values = values[mask]
    assert len(np.unique(indices)) == len(unique_values)
    return unique_values


# def _compress_unique_values(orig_val, col, indices):
    # values = np.take(orig_val, col, axis=1)
    # values = values.reshape(-1, 1) if orig_val.ndim == 1 else values
    # indices = indices.reshape(-1, 1)
    # unique_indices = np.unique(indices)
    # indices_plus_values = np.concatenate([indices, values], axis=-1)
    #   np.unique(axis=0) will sort the data based on first column,
    #   so we can do direct mapping, then remove redundant unique_indices
    # unique_values = np.unique(indices_plus_values, axis=0)
    # diff = True if len(unique_indices) != len(unique_values) else False
    # if diff:
    #    print(colorize("some users or items contain different features, "
    #                   "will only keep the last one", "red"))
    #    mask = np.concatenate([unique_values[:-1, 0] != unique_values[1:, 0],
    #                           np.array([True])])
    #    unique_values = unique_values[mask]

    # assert len(unique_indices) == len(unique_values)
    # return unique_values[:, 1:]


def get_predict_indices_and_values(data_info, user, item, n_items,
                                   sparse, dense):
    if isinstance(user, numbers.Integral):
        user = list([user])
    if isinstance(item, numbers.Integral):
        item = list([item])

    sparse_indices = get_sparse_indices(
        data_info, user, item, mode="predict"
    ) if sparse else None
    dense_values = get_dense_values(
        data_info, user, item, mode="predict"
    ) if dense else None
    if sparse and dense:
        assert len(sparse_indices) == len(dense_values), (
            "indices and values length must equal"
        )
    return user, item, sparse_indices, dense_values


def get_recommend_indices_and_values(data_info, user, n_items, sparse, dense):
    user_indices = np.repeat(user, n_items)
    item_indices = np.arange(n_items)

    sparse_indices = get_sparse_indices(
        data_info, user, n_items=n_items, mode="recommend"
    ) if sparse else None
    dense_values = get_dense_values(
        data_info, user, n_items=n_items, mode="recommend"
    ) if dense else None
    if sparse and dense:
        assert len(sparse_indices) == len(dense_values), (
            "indices and values length must equal")

    return user_indices, item_indices, sparse_indices, dense_values


def get_sparse_indices(data_info, user, item=None, n_items=None,
                       mode="predict"):
    user_sparse_col = data_info.user_sparse_col.index
    item_sparse_col = data_info.item_sparse_col.index
    orig_cols = user_sparse_col + item_sparse_col
    # keep column names in original order
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]

    if mode == "predict":
        if user_sparse_col and item_sparse_col:
            user_sparse_part = data_info.user_sparse_unique[user]
            item_sparse_part = data_info.item_sparse_unique[item]
            sparse_indices = np.concatenate(
                [user_sparse_part, item_sparse_part], axis=-1)[:, col_reindex]
            return sparse_indices
        elif user_sparse_col:
            return data_info.user_sparse_unique[user]
        elif item_sparse_col:
            return data_info.item_sparse_unique[item]

    elif mode == "recommend":
        if user_sparse_col and item_sparse_col:
            user_sparse_part = np.tile(data_info.user_sparse_unique[user],
                                       (n_items, 1))
            item_sparse_part = data_info.item_sparse_unique[:-1]  # remove oov
            sparse_indices = np.concatenate(
                [user_sparse_part, item_sparse_part], axis=-1)[:, col_reindex]
            return sparse_indices
        elif user_sparse_col:
            return np.tile(data_info.user_sparse_unique[user], (n_items, 1))
        elif item_sparse_col:
            return data_info.item_sparse_unique[:-1]


def get_dense_indices(data_info, user, n_items=None, mode="predict"):
    user_dense_col = data_info.user_dense_col.index
    item_dense_col = data_info.item_dense_col.index
    total_dense_cols = len(user_dense_col) + len(item_dense_col)
    if mode == "predict":
        return np.tile(np.arange(total_dense_cols), (len(user), 1))
    elif mode == "recommend":
        return np.tile(np.arange(total_dense_cols), (n_items, 1))


def get_dense_values(data_info, user, item=None, n_items=None, mode="predict"):
    user_dense_col = data_info.user_dense_col.index
    item_dense_col = data_info.item_dense_col.index
    # keep column names in original order
    orig_cols = user_dense_col + item_dense_col
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]

    if mode == "predict":
        if user_dense_col and item_dense_col:
            user_dense_part = data_info.user_dense_unique[user]
            item_dense_part = data_info.item_dense_unique[item]
            dense_values = np.concatenate(
                [user_dense_part, item_dense_part], axis=-1)[:, col_reindex]
            return dense_values
        elif user_dense_col:
            return data_info.user_dense_unique[user]
        elif item_dense_col:
            return data_info.item_dense_unique[item]

    elif mode == "recommend":
        if user_dense_col and item_dense_col:
            user_dense_part = np.tile(data_info.user_dense_unique[user],
                                      (n_items, 1))
            item_dense_part = data_info.item_dense_unique[:-1]  # remove oov
            dense_values = np.concatenate(
                [user_dense_part, item_dense_part], axis=-1)[:, col_reindex]
            return dense_values
        elif user_dense_col:
            return np.tile(data_info.user_dense_unique[user], (n_items, 1))
        elif item_dense_col:
            return data_info.item_dense_unique[:-1]


# This function will try not to modify the original numpy arrays
def features_from_dict(data_info, sparse_indices, dense_values, feats, mode):
    if mode == "predict":
        # data_info.col_name_mapping: {"sparse_col": {col: index}}
        sparse_mapping = data_info.col_name_mapping["sparse_col"]
        dense_mapping = data_info.col_name_mapping["dense_col"]
    elif mode == "recommend":
        # in recommend scenario will only change user features
        sparse_mapping = data_info.col_name_mapping["user_sparse_col"]
        dense_mapping = data_info.col_name_mapping["user_dense_col"]
    else:
        raise ValueError("mode must be predict or recommend")

    sparse_indices_copy = (None if sparse_indices is None
                           else sparse_indices.copy())
    dense_values_copy = (None if dense_values is None
                         else dense_values.copy())
    for col, val in feats.items():
        if sparse_indices is not None and col in sparse_mapping:
            field_idx = sparse_mapping[col]
            if val in data_info.sparse_unique_idxs[col]:
                # data_info.sparse_unique_idxs: {col: {value: index}}
                feat_idx = data_info.sparse_unique_idxs[col][val]
                offset = data_info.sparse_offset[field_idx]
                sparse_indices_copy[:, field_idx] = feat_idx + offset
            # else:
                # if val not exists, assign to oov position
            #    sparse_indices_copy[:, field_idx] = (
            #        data_info.sparse_oov[field_idx]
            #    )
        elif dense_values is not None and col in dense_mapping:
            field_idx = dense_mapping[col]
            dense_values_copy[:, field_idx] = val

    return sparse_indices_copy, dense_values_copy


def features_from_batch_data(data_info, sparse, dense, data):
    if sparse:
        sparse_col_num = len(data_info.col_name_mapping["sparse_col"])
        sparse_indices = [_ for _ in range(sparse_col_num)]
        for col, field_idx in data_info.col_name_mapping["sparse_col"].items():
            if col not in data.columns:
                continue
            sparse_indices[field_idx] = compute_sparse_feat_indices(
                data_info, data, field_idx, col)
        sparse_indices = np.array(sparse_indices).astype(np.int64).T
    else:
        sparse_indices = None

    if dense:
        dense_col_num = len(data_info.col_name_mapping["dense_col"])
        dense_values = [_ for _ in range(dense_col_num)]
        for col, field_idx in data_info.col_name_mapping["dense_col"].items():
            if col not in data.columns:
                continue
            dense_values[field_idx] = data[col].to_numpy()
        dense_values = np.array(dense_values).T
    else:
        dense_values = None

    if sparse and dense:
        assert len(sparse_indices) == len(dense_values), (
            "indices and values length must equal"
        )
    return sparse_indices, dense_values


# This function will try not to modify the original numpy arrays
def add_item_features(data_info, sparse_indices, dense_values, data):
    data = _check_oov(data_info, data, "item")
    row_idx = data["item"].to_numpy()

    sparse_indices_copy = (None if sparse_indices is None
                           else sparse_indices.copy())
    col_info = data_info.item_sparse_col
    if sparse_indices is not None and col_info.name:
        sparse_indices_copy = sparse_indices.copy()
        for feat_idx, col in enumerate(col_info.name):
            if col not in data.columns:
                continue
            sparse_indices_copy[row_idx, feat_idx] = (
                compute_sparse_feat_indices(
                    data_info, data, col_info.index[feat_idx], col)
            )

    dense_values_copy = (None if dense_values is None
                         else dense_values.copy())
    col_info = data_info.item_dense_col
    if dense_values is not None and col_info.name:
        dense_values_copy = dense_values.copy()
        for feat_idx, col in enumerate(col_info.name):
            if col not in data.columns:
                continue
            dense_values_copy[row_idx, feat_idx] = data[col].to_numpy()
    return sparse_indices_copy, dense_values_copy


def compute_sparse_feat_indices(data_info, data, field_idx, column):
    offset = data_info.sparse_offset[field_idx]
    oov_val = data_info.sparse_oov[field_idx]
    map_vals = data_info.sparse_unique_idxs[column]
    values = data[column].tolist()
    feat_indices = np.array(
        [map_vals[v] + offset if v in map_vals else oov_val
         for v in values]
    )
    return feat_indices


# This function will try not to modify the original data
def _check_oov(data_info, orig_data, mode):
    data = orig_data.copy()
    if mode == "user":
        users = data.user.tolist()
        user_mapping = data_info.user2id
        user_ids = [user_mapping[u] if u in user_mapping else -1 for u in users]
        data["user"] = user_ids
        data = data[data.user != -1]
    elif mode == "item":
        items = data.item.tolist()
        item_mapping = data_info.item2id
        item_ids = [item_mapping[i] if i in item_mapping else -1 for i in items]
        data["item"] = item_ids
        data = data[data.item != -1]
    return data
