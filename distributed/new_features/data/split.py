import math
import numpy as np
from sklearn.model_selection import train_test_split


def random_split(data, test_size=0.2, multi_ratios=None, shuffle=True, seed=42):
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)
    if not isinstance(ratios, list):
        ratios = list(ratios)

    split_data_all = []
    for i in range(n_splits - 1):
        size = ratios.pop(-1)
        ratios = [r / math.fsum(ratios) for r in ratios]
        data, split_data = train_test_split(data, test_size=size, shuffle=shuffle, random_state=seed)
        split_data_all.insert(0, split_data)
    split_data_all.insert(0, data)
    return split_data_all


def _groupby_user(user_indices, order):
    sort_kind = "mergesort" if order else "quicksort"
    users, user_position, user_counts = np.unique(user_indices, return_inverse=True, return_counts=True)
    user_split_indices = np.split(np.argsort(user_position, kind=sort_kind),
                                  np.cumsum(user_counts)[:-1])
    return user_split_indices


def _check_and_convert_ratio(test_size, multi_ratios):
    if test_size is not None:
        assert isinstance(test_size, float), "test_size must be float value"
        assert 0 < test_size < 1, "test_size must be in (0.0, 1.0)"
        ratios = [1 - test_size, test_size]
        return ratios, 2

    elif isinstance(multi_ratios, (list, tuple)):
        assert len(multi_ratios) > 1, "multi_ratios must at least have two elements"
        assert all([r > 0.0 for r in multi_ratios]), "ratios should be positive values"
        if math.fsum(multi_ratios) != 1.0:
            ratios = [r / math.fsum(multi_ratios) for r in multi_ratios]
        else:
            ratios = multi_ratios
        return ratios, len(ratios)

    else:
        raise ValueError("multi_ratios should be list or tuple")


def split_by_ratio(data, order=True, shuffle=False, test_size=0.2, multi_ratios=None, seed=42):
    np.random.seed(seed)
    assert ("user" in data.columns), "data must contains user column"
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)

    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    cum_ratios = np.cumsum(ratios).tolist()[:-1]
    split_indices_all = [[] for _ in range(n_splits)]
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:
            split_indices_all[0].extend(u_data)
        else:
            u_split_data = np.split(u_data, [round(cum * u_data_len) for cum in cum_ratios])
            for i in range(n_splits):
                split_indices_all[i].extend(u_split_data[i])

    if shuffle:
        return [np.random.permutation(data[idx]) for idx in split_indices_all]
    return [data[idx] for idx in split_indices_all]


def split_by_num(data, order=True, shuffle=False, test_size=1, seed=42):
    np.random.seed(seed)
    assert isinstance(test_size, int), "test_size must be int value"
    assert 0 < test_size < len(data), "test_size must be in (0, len(data))"

    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    train_indices = []
    test_indices = []
    for u in n_users:
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:  # keep rare users in trainset
            train_indices.extend(u_data)
        elif u_data_len <= test_size:
            train_indices.extend(u_data[:-1])
            test_indices.extend(u_data[-1])
        else:
            k = test_size
            train_indices.extend(u_data[:-k])
            test_indices.extend(u_data[-k:])

    if shuffle:
        train_indices = np.random.permutation(train_indices)
        test_indices = np.random.permutation(test_indices)
    return data[train_indices], data[test_indices]


def split_by_ratio_chrono(data):
    data.sort_values(by=["timestamp"], inplace=True)


def split_by_num_chrono(data):
    pass

