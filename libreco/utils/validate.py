import numpy as np
import pandas as pd

from ..utils.exception import NotSamplingError
from ..utils.misc import colorize


def convert_id(model, user, item, inner_id=False):
    if not isinstance(user, (list, tuple, np.ndarray, pd.Series)):
        user = [user]
    if not isinstance(item, (list, tuple, np.ndarray, pd.Series)):
        item = [item]
    if not inner_id:
        user = [model.data_info.user2id.get(u, model.n_users) for u in user]
        item = [model.data_info.item2id.get(i, model.n_items) for i in item]
    return np.array(user), np.array(item)


def check_unknown(model, user, item):
    unknown_user_indices = list(np.where(user == model.n_users)[0])
    unknown_item_indices = list(np.where(item == model.n_items)[0])
    unknown_index = list(set(unknown_user_indices) | set(unknown_item_indices))
    unknown_num = len(unknown_index)
    if unknown_num > 0:
        unknown_str = (
            f"Detect {unknown_num} unknown interaction(s), "
            f"position: {unknown_index}"
        )
        print(f"{colorize(unknown_str, 'red')}")
    return unknown_num, unknown_index, user, item


def check_unknown_user(model, user, inner_id=False):
    user_id = model.data_info.user2id.get(user, -1) if not inner_id else user
    if 0 <= user_id < model.n_users:
        return user_id
    else:
        if not inner_id:
            unknown_str = f"detect unknown user: {user}"
            print(f"{colorize(unknown_str, 'red')}")
        return


def check_has_sampled(data, verbose):
    if not data.has_sampled and verbose > 1:
        exception_str = (
            "During training, "
            "one must do whole data sampling "
            "before evaluating on epochs."
        )
        raise NotSamplingError(f"{colorize(exception_str, 'red')}")


def check_interaction_mode(recent_num, random_num):
    if recent_num is not None:
        assert isinstance(recent_num, int), "recent_num must be integer"
        mode = "recent"
        num = recent_num
    elif random_num is not None:
        assert isinstance(random_num, int), "random_num must be integer"
        mode = "random"
        num = random_num
    else:
        mode = "recent"
        num = 10  # by default choose 10 recent interactions
    return mode, num


def check_sparse_indices(data_info):
    return False if not data_info.sparse_col.name else True


def check_dense_values(data_info):
    return False if not data_info.dense_col.name else True


def sparse_feat_size(data_info):
    if (
        data_info.user_sparse_unique is not None
        and data_info.item_sparse_unique is not None
    ):
        return (
            max(
                np.max(data_info.user_sparse_unique),
                np.max(data_info.item_sparse_unique),
            )
            + 1
        )
    elif data_info.user_sparse_unique is not None:
        return np.max(data_info.user_sparse_unique) + 1
    elif data_info.item_sparse_unique is not None:
        return np.max(data_info.item_sparse_unique) + 1


def sparse_field_size(data_info):
    return len(data_info.sparse_col.name)


def dense_field_size(data_info):
    return len(data_info.dense_col.name)


def check_multi_sparse(data_info, multi_sparse_combiner):
    if data_info.multi_sparse_combine_info and multi_sparse_combiner is not None:
        if multi_sparse_combiner not in ("normal", "sum", "mean", "sqrtn"):
            raise ValueError(
                f"unsupported multi_sparse_combiner type: " f"{multi_sparse_combiner}"
            )
        else:
            combiner = multi_sparse_combiner
    else:
        combiner = "normal"
    return combiner


def true_sparse_field_size(data_info, sparse_field_size, combiner):
    # When using multi_sparse_combiner, field size will decrease.
    if data_info.multi_sparse_combine_info and combiner in ("sum", "mean", "sqrtn"):
        field_length = data_info.multi_sparse_combine_info.field_len
        return sparse_field_size - (sum(field_length) - len(field_length))
    else:
        return sparse_field_size
