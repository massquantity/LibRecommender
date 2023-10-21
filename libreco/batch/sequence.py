import random

import numpy as np


def get_sparse_interacted(user_indices, item_indices, user_consumed, mode, num, np_rng):
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
            chosen_items = np_rng.choice(consumed_items, num, replace=False)
            interacted_items.extend(chosen_items.tolist())

    interacted_indices = np.asarray(interacted_indices).reshape(-1, 1)
    indices = np.concatenate(
        [interacted_indices, np.zeros_like(interacted_indices)], axis=1
    )
    return indices, np.array(interacted_items), len(user_indices)


def get_interacted_seqs(
    user_indices,
    item_indices,
    user_consumed,
    pad_index,
    mode,
    max_seq_len,
    user_consumed_set,
    np_rng,
):
    batch_size = len(user_indices)
    seqs = np.full((batch_size, max_seq_len), pad_index, dtype=np.int32)
    seq_lens = []
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
            seq_lens.append(1)
        elif position < max_seq_len:
            seqs[j, :position] = consumed_items[:position]
            seq_lens.append(position)
        else:
            if mode == "recent":
                start_index = position - max_seq_len
                seqs[j] = consumed_items[start_index:position]
            elif mode == "random":
                chosen_items = np_rng.choice(consumed_items, max_seq_len, replace=False)
                seqs[j] = chosen_items
            seq_lens.append(max_seq_len)

    return seqs, np.array(seq_lens, dtype=np.int32)


# most recent num items a user has interacted, assume already sorted by time.
def get_recent_seqs(n_users, user_consumed, pad_index, max_seq_len):
    recent_seqs = np.full((n_users, max_seq_len), pad_index, dtype=np.int32)
    recent_seq_lens = []
    for u in range(n_users):
        u_consumed_items = user_consumed[u]
        u_items_len = len(u_consumed_items)
        if u_items_len < max_seq_len:
            recent_seqs[u, :u_items_len] = u_consumed_items
            recent_seq_lens.append(u_items_len)
        else:
            recent_seqs[u] = u_consumed_items[-max_seq_len:]
            recent_seq_lens.append(max_seq_len)

    oov = np.full(max_seq_len, pad_index, dtype=np.int32)
    recent_seqs = np.vstack([recent_seqs, oov])
    recent_seq_lens = np.append(recent_seq_lens, [1])
    return recent_seqs, np.array(recent_seq_lens, dtype=np.int32)


def get_dual_seqs(
    user_indices,
    item_indices,
    user_consumed,
    pad_index,
    long_max_len,
    short_max_len,
    user_consumed_set,
):
    batch_size = len(user_indices)
    long_seqs = np.full((batch_size, long_max_len), pad_index, dtype=np.int32)
    short_seqs = np.full((batch_size, short_max_len), pad_index, dtype=np.int32)
    long_seq_lens, short_seq_lens = [], []
    total_max_len = long_max_len + short_max_len
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
            long_seq_lens.append(1)
            short_seq_lens.append(1)
        elif position <= short_max_len:
            long_seq_lens.append(1)
            if position < short_max_len:
                short_seqs[j, :position] = consumed_items[:position]
            else:
                short_seqs[j] = consumed_items[:position]
            short_seq_lens.append(position)
        else:
            if position < total_max_len:
                long_size = position - short_max_len
                long_seqs[j, :long_size] = consumed_items[:long_size]
                long_seq_lens.append(long_size)
            else:
                long_start = position - total_max_len
                long_seqs[j] = consumed_items[long_start : long_start + long_max_len]
                long_seq_lens.append(long_max_len)
            short_start = position - short_max_len
            short_seqs[j] = consumed_items[short_start:position]
            short_seq_lens.append(short_max_len)

    return (
        long_seqs,
        np.array(long_seq_lens, dtype=np.int32),
        short_seqs,
        np.array(short_seq_lens, dtype=np.int32),
    )


def get_recent_dual_seqs(
    n_users, user_consumed, pad_index, long_max_len, short_max_len
):
    long_seqs = np.full((n_users, long_max_len), pad_index, dtype=np.int32)
    short_seqs = np.full((n_users, short_max_len), pad_index, dtype=np.int32)
    long_seq_lens, short_seq_lens = [], []
    total_max_len = long_max_len + short_max_len
    for u in range(n_users):
        consumed_items = user_consumed[u]
        items_len = len(consumed_items)
        if items_len <= short_max_len:
            long_seq_lens.append(1)
            short_seqs[u, :items_len] = consumed_items[:items_len]
            short_seq_lens.append(items_len)
        else:
            if items_len < total_max_len:
                long_size = items_len - short_max_len
                long_seqs[u, :long_size] = consumed_items[:long_size]
                long_seq_lens.append(long_size)
            else:
                long_start = items_len - total_max_len
                long_seqs[u] = consumed_items[long_start : long_start + long_max_len]
                long_seq_lens.append(long_max_len)
            short_start = items_len - short_max_len
            short_seqs[u] = consumed_items[short_start:]
            short_seq_lens.append(short_max_len)

    long_oov = np.full(long_max_len, pad_index, dtype=np.int32)
    long_seqs = np.vstack([long_seqs, long_oov])
    long_seq_lens = np.append(long_seq_lens, [1])
    short_oov = np.full(short_max_len, pad_index, dtype=np.int32)
    short_seqs = np.vstack([short_seqs, short_oov])
    short_seq_lens = np.append(short_seq_lens, [1])
    return (
        long_seqs,
        np.array(long_seq_lens, dtype=np.int32),
        short_seqs,
        np.array(short_seq_lens, dtype=np.int32),
    )
