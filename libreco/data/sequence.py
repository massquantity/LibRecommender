from math import floor
from random import random

import numpy as np

from ..tfops import tf


def sparse_user_interacted(
    user_indices, item_indices, user_consumed, mode=None, num=None
):
    interacted_indices = []
    interacted_items = []
    for j, (u, i) in enumerate(zip(user_indices, item_indices)):
        consumed_items = user_consumed[u]
        position = consumed_items.index(i)
        if position == 0:  # first item, no history interaction
            continue
        elif position < num:
            interacted_indices.extend([j] * position)
            interacted_items.extend(consumed_items[:position])
        elif position >= num and mode == "recent":
            start_index = position - num
            interacted_indices.extend([j] * num)
            interacted_items.extend(consumed_items[start_index:position])
        elif position >= num and mode == "random":
            interacted_indices.extend([j] * num)
            chosen_items = np.random.choice(consumed_items, num, replace=False)
            interacted_items.extend(chosen_items.tolist())

    assert len(interacted_indices) == len(
        interacted_items
    ), "length of indices and values doesn't match"
    interacted_indices = np.asarray(interacted_indices).reshape(-1, 1)
    indices = np.concatenate(
        [interacted_indices, np.zeros_like(interacted_indices)], axis=1
    )
    return indices, interacted_items, len(user_indices)


# used in YoutubeRetrieval
def sparse_user_last_interacted(n_users, user_consumed, recent_num=10):
    assert isinstance(recent_num, int), "recent_num must be integer"
    interacted_indices = []
    interacted_items = []
    for u in range(n_users):
        u_consumed_items = user_consumed[u]
        u_items_len = len(u_consumed_items)
        if u_items_len < recent_num:
            interacted_indices.extend([u] * u_items_len)
            interacted_items.extend(u_consumed_items)
        else:
            interacted_indices.extend([u] * recent_num)
            interacted_items.extend(u_consumed_items[-recent_num:])

    assert len(interacted_indices) == len(
        interacted_items
    ), "length of indices and values doesn't match"
    interacted_indices = np.asarray(interacted_indices).reshape(-1, 1)
    indices = np.concatenate(
        [interacted_indices, np.zeros_like(interacted_indices)], axis=1
    )
    return indices, interacted_items


def sample_item_with_tolerance(num, consumed_items, consumed_len, tolerance=5):
    assert num > tolerance
    sampled = []
    first_len = num - tolerance
    while len(sampled) < first_len:
        i = floor(random() * consumed_len)
        if consumed_items[i] not in sampled:
            sampled.append(consumed_items[i])
    for _ in range(tolerance):
        i = floor(random() * consumed_len)
        sampled.append(consumed_items[i])
    return sampled


def user_interacted_seq(
    user_indices,
    item_indices,
    user_consumed,
    pad_index,
    mode=None,
    num=None,
    user_consumed_set=None,
):
    batch_size = len(user_indices)
    batch_interacted = np.full((batch_size, num), pad_index, dtype=np.int32)
    batch_interacted_len = []
    for j, (u, i) in enumerate(zip(user_indices, item_indices)):
        consumed_items = user_consumed[u]
        consumed_len = len(consumed_items)
        consumed_set = user_consumed_set[u]
        # If `i` is a negative item, then random sample some items
        # from user's past interacted items.
        # TODO: sample sequence from user past interactions
        if i not in consumed_set:
            if consumed_len >= num:
                # `np.random.choice` is too slow,
                # so here we use a custom sample function with
                # some tolerance of duplicate items
                chosen_items = sample_item_with_tolerance(
                    num, consumed_items, consumed_len, 5
                )
                batch_interacted[j] = chosen_items
                batch_interacted_len.append(float(num))
            else:
                batch_interacted[j, :consumed_len] = consumed_items
                batch_interacted_len.append(float(consumed_len))
        else:
            position = consumed_items.index(i)
            if position == 0:
                # first item, no historical interaction,
                # assign to pad_index by default, and length is 1.
                batch_interacted_len.append(1.0)
            elif position < num:
                batch_interacted[j, :position] = consumed_items[:position]
                batch_interacted_len.append(float(position))
            elif position >= num and mode == "recent":
                start_index = position - num
                batch_interacted[j] = consumed_items[start_index:position]
                batch_interacted_len.append(float(num))
            elif position >= num and mode == "random":
                chosen_items = np.random.choice(consumed_items, num, replace=False)
                batch_interacted[j] = chosen_items
                batch_interacted_len.append(float(num))

    return batch_interacted, batch_interacted_len


# most recent num items a user has interacted, assume already sorted by time.
def get_user_last_interacted(n_users, user_consumed, pad_index, recent_num=10):
    u_last_interacted = np.full((n_users, recent_num), pad_index, dtype=np.int32)
    interacted_len = []
    for u in range(n_users):
        u_consumed_items = user_consumed[u]
        u_items_len = len(u_consumed_items)
        if u_items_len < recent_num:
            u_last_interacted[u, :u_items_len] = u_consumed_items
            interacted_len.append(float(u_items_len))
        else:
            u_last_interacted[u] = u_consumed_items[-recent_num:]
            interacted_len.append(float(recent_num))

    return u_last_interacted, np.asarray(interacted_len)


# used in SVD++
def sparse_tensor_interaction(data, recent_num=None, random_sample_rate=None):
    sparse_data = data.sparse_interaction.tocoo()
    row = sparse_data.row.reshape(-1, 1)
    indices = np.concatenate([row, np.zeros_like(row)], axis=1)
    values = sparse_data.col

    # user_interacted_num = np.diff(data.sparse_interaction.indptr)
    if recent_num is not None:
        indices, values = user_recent_interact(recent_num, indices, values)
    elif random_sample_rate is not None:
        indices, values = random_sample(random_sample_rate, indices, values)

    sparse_tensor = tf.SparseTensor(
        indices=indices, values=values, dense_shape=sparse_data.shape
    )
    return sparse_tensor


def random_sample(sample_rate, indices, values):
    assert 0.0 < sample_rate < 1.0, "sample_rate must be in (0.0, 1.0)"
    total_length = len(values)
    sample_num = int(total_length * sample_rate)
    sampled_indices = np.random.choice(
        range(total_length), size=sample_num, replace=False
    )
    indices = indices[sampled_indices]
    values = values[sampled_indices]
    return indices, values


def user_recent_interact(num, indices, values):
    assert isinstance(num, int), "recent_interact_num must be int"
    users, user_position, user_counts = np.unique(
        indices[:, 0], return_inverse=True, return_counts=True
    )
    user_split_indices = np.split(
        np.argsort(user_position, kind="mergesort"), np.cumsum(user_counts)[:-1]
    )

    n_users = len(users)
    recent_indices = list()
    for u in range(n_users):
        # assume user interactions have already been sorted by time.
        u_data = user_split_indices[u][-num:]
        recent_indices.extend(u_data)
    indices = indices[recent_indices]
    values = values[recent_indices]
    return indices, values
