import sys
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd


def interaction_consumed(user_indices, item_indices):
    user_consumed = defaultdict(list)
    item_consumed = defaultdict(list)
    for u, i in zip(user_indices, item_indices):
        if isinstance(u, np.integer):
            u = u.item()
        if isinstance(i, np.integer):
            i = i.item()
        user_consumed[u].append(i)
        item_consumed[i].append(u)
    return _remove_duplicates(user_consumed, item_consumed)


def _remove_duplicates(user_consumed, item_consumed):
    # keys will preserve order in dict since Python3.7
    if sys.version_info[:2] >= (3, 7):
        dict_func = dict.fromkeys
    else:  # pragma: no cover
        dict_func = OrderedDict.fromkeys
    user_dedup = {u: list(dict_func(items)) for u, items in user_consumed.items()}
    item_dedup = {i: list(dict_func(users)) for i, users in item_consumed.items()}
    return user_dedup, item_dedup


def update_consumed(new_data_info, data_info, merge_behavior):
    if merge_behavior:
        new_data_info.user_consumed = _merge_dedup(
            new_data_info.user_consumed, new_data_info.n_users, data_info.user_consumed
        )
        new_data_info.item_consumed = _merge_dedup(
            new_data_info.item_consumed, new_data_info.n_items, data_info.item_consumed
        )
    else:
        new_data_info.user_consumed = _fill_empty(
            new_data_info.user_consumed, new_data_info.n_users, data_info.user_consumed
        )
        new_data_info.item_consumed = _fill_empty(
            new_data_info.item_consumed, new_data_info.n_items, data_info.item_consumed
        )
    return new_data_info


def _merge_dedup(new_consumed, num, old_consumed):
    result = dict()
    for i in range(num):
        assert i in new_consumed or i in old_consumed
        if i in new_consumed and i in old_consumed:
            consumed = old_consumed[i] + new_consumed[i]
            result[i] = _remove_first_duplicates(consumed)
        else:
            result[i] = new_consumed[i] if i in new_consumed else old_consumed[i]
    return result


# some users may not appear in new data
def _fill_empty(consumed, num, old_consumed):
    return {i: consumed[i] if i in consumed else old_consumed[i] for i in range(num)}


def _remove_first_duplicates(consumed):
    return pd.Series(consumed).drop_duplicates(keep="last").tolist()
