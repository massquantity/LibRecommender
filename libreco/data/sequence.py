import random

import numpy as np


def get_sparse_interacted(
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

    interacted_indices = np.asarray(interacted_indices).reshape(-1, 1)
    indices = np.concatenate(
        [interacted_indices, np.zeros_like(interacted_indices)], axis=1
    )
    return indices, np.array(interacted_items), len(user_indices)


def get_interacted_seq(
    user_indices,
    item_indices,
    user_consumed,
    pad_index,
    mode,
    num,
    user_consumed_set,
    np_rng,
):
    batch_size = len(user_indices)
    batch_interacted = np.full((batch_size, num), pad_index, dtype=np.int32)
    batch_interacted_len = []
    for j, (u, i) in enumerate(zip(user_indices, item_indices)):
        consumed_items = user_consumed[u]
        consumed_len = len(consumed_items)
        consumed_set = user_consumed_set[u]
        # If `i` is a negative item, sample sequence from user's past interaction
        position = (
            consumed_items.index(i)
            if i in consumed_set
            else random.randrange(0, consumed_len)
        )
        if position == 0:
            # first item has no historical interaction, fill in with pad_index
            batch_interacted_len.append(1.0)
        elif position < num:
            batch_interacted[j, -position:] = consumed_items[:position]
            batch_interacted_len.append(float(position))
        else:
            if mode == "recent":
                start_index = position - num
                batch_interacted[j] = consumed_items[start_index:position]
            elif mode == "random":
                chosen_items = np_rng.choice(consumed_items, num, replace=False)
                batch_interacted[j] = chosen_items
            batch_interacted_len.append(float(num))

    return batch_interacted, np.array(batch_interacted_len)


# most recent num items a user has interacted, assume already sorted by time.
def get_user_last_interacted(n_users, user_consumed, pad_index, recent_num=10):
    u_last_interacted = np.full((n_users, recent_num), pad_index, dtype=np.int32)
    interacted_len = []
    for u in range(n_users):
        u_consumed_items = user_consumed[u]
        u_items_len = len(u_consumed_items)
        if u_items_len < recent_num:
            u_last_interacted[u, -u_items_len:] = u_consumed_items
            interacted_len.append(float(u_items_len))
        else:
            u_last_interacted[u] = u_consumed_items[-recent_num:]
            interacted_len.append(float(recent_num))

    return u_last_interacted, np.array(interacted_len)
