import math
import random

import numpy as np


def _check_invalid_negatives(negatives, items_pos, items=None):
    if items is not None and len(items) > 0:
        invalid_indices = np.union1d(
            np.where(negatives == items_pos)[0], np.where(negatives == items)[0]
        )
    else:
        invalid_indices = np.where(negatives == items_pos)[0]
    return list(invalid_indices)


def negatives_from_random(
    np_rng, n_items, items_pos, num_neg, items=None, tolerance=10
):
    items_pos = np.repeat(items_pos, num_neg) if num_neg > 1 else items_pos
    items = np.repeat(items, num_neg) if num_neg > 1 and items is not None else items
    replace = False if len(items_pos) < n_items else True
    negatives = np_rng.choice(n_items, size=len(items_pos), replace=replace)
    invalid_indices = _check_invalid_negatives(negatives, items_pos, items)
    if invalid_indices:
        for _ in range(tolerance):
            negatives[invalid_indices] = np_rng.choice(
                n_items, size=len(invalid_indices), replace=True
            )
            invalid_indices = _check_invalid_negatives(negatives, items_pos, items)
            if not invalid_indices:
                break
    return negatives


def negatives_from_popular(np_rng, n_items, items_pos, num_neg, items=None, probs=None):
    items_pos = np.repeat(items_pos, num_neg) if num_neg > 1 else items_pos
    items = np.repeat(items, num_neg) if num_neg > 1 and items is not None else items
    negatives = np_rng.choice(n_items, size=len(items_pos), replace=True, p=probs)
    invalid_indices = _check_invalid_negatives(negatives, items_pos, items)
    if invalid_indices:
        negatives[invalid_indices] = np_rng.choice(
            n_items, size=len(invalid_indices), replace=True, p=probs
        )
    return negatives


def negatives_from_out_batch(np_rng, n_items, items_pos, items, num_neg):
    sample_num = len(items_pos) * num_neg
    candidate_items = list(set(range(n_items)) - set(items_pos) - set(items))
    if not candidate_items:
        return np_rng.choice(n_items, size=sample_num, replace=True)
    replace = False if sample_num < len(candidate_items) else True
    return np_rng.choice(candidate_items, size=sample_num, replace=replace)


def negatives_from_unconsumed(
    user_consumed_set, users, items, n_items, num_neg, tolerance=10
):
    _floor = math.floor
    _random = random.random

    def sample_one():
        return _floor(n_items * _random())

    negatives = []
    for u, i in zip(users, items):
        u_negs = []
        for _ in range(num_neg):
            success = False
            for _ in range(tolerance):
                n = sample_one()
                if n != i and n not in u_negs and n not in user_consumed_set[u]:
                    success = True
                    break
            if not success:
                for _ in range(tolerance):
                    n = sample_one()
                    if n != i and n not in u_negs:
                        success = True
                        break
            if not success:
                n = sample_one()
                print(f"possible not enough negatives for user {u} and item {i}.")
            # noinspection PyUnboundLocalVariable
            u_negs.append(n)
        negatives.extend(u_negs)
    return np.array(negatives)


def neg_probs_from_frequency(item_consumed, n_items, temperature):
    freqs = []
    for i in range(n_items):
        freq = len(set(item_consumed[i]))
        if temperature != 1.0:
            freq = pow(freq, temperature)
        freqs.append(freq)
    freqs = np.array(freqs)
    return freqs / np.sum(freqs)


def pos_probs_from_frequency(item_consumed, n_users, n_items, alpha):
    probs = []
    for i in range(n_items):
        prob = len(set(item_consumed[i])) / n_users
        prob = (math.sqrt(prob / alpha) + 1) * (alpha / prob)
        probs.append(prob)
    return probs
