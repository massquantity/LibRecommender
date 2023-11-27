import sys
from collections import OrderedDict, defaultdict

import numpy as np


def interaction_consumed(user_indices, item_indices):
    """The underlying rust function will remove consecutive repeated elements."""
    if isinstance(user_indices, np.ndarray):
        user_indices = user_indices.tolist()
    if isinstance(item_indices, np.ndarray):
        item_indices = item_indices.tolist()

    try:
        from recfarm import build_consumed_unique

        return build_consumed_unique(user_indices, item_indices)
    except ModuleNotFoundError:  # pragma: no cover
        return _interaction_consumed(user_indices, item_indices)


def _interaction_consumed(user_indices, item_indices):  # pragma: no cover
    user_consumed = defaultdict(list)
    item_consumed = defaultdict(list)
    for u, i in zip(user_indices, item_indices):
        user_consumed[u].append(i)
        item_consumed[i].append(u)
    return _remove_duplicates(user_consumed, item_consumed)


def _remove_duplicates(user_consumed, item_consumed):  # pragma: no cover
    # keys will preserve order in dict since Python3.7
    if sys.version_info[:2] >= (3, 7):
        dict_func = dict.fromkeys
    else:  # pragma: no cover
        dict_func = OrderedDict.fromkeys
    user_dedup = {u: list(dict_func(items)) for u, items in user_consumed.items()}
    item_dedup = {i: list(dict_func(users)) for i, users in item_consumed.items()}
    return user_dedup, item_dedup


def update_consumed(
    user_indices, item_indices, n_users, n_items, old_info, merge_behavior
):
    user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
    if merge_behavior:
        user_consumed = _merge_dedup(user_consumed, n_users, old_info.user_consumed)
        item_consumed = _merge_dedup(item_consumed, n_items, old_info.item_consumed)
    else:
        user_consumed = _fill_empty(user_consumed, n_users, old_info.user_consumed)
        item_consumed = _fill_empty(item_consumed, n_items, old_info.item_consumed)
    return user_consumed, item_consumed


def _merge_dedup(new_consumed, num, old_consumed):
    result = dict()
    for i in range(num):
        assert i in new_consumed or i in old_consumed
        if i in new_consumed and i in old_consumed:
            result[i] = old_consumed[i] + new_consumed[i]
        else:
            result[i] = new_consumed[i] if i in new_consumed else old_consumed[i]
    return result


# some users may not appear in new data
def _fill_empty(consumed, num, old_consumed):
    return {i: consumed[i] if i in consumed else old_consumed[i] for i in range(num)}


# def _remove_first_duplicates(consumed):
#    return pd.Series(consumed).drop_duplicates(keep="last").tolist()
