import itertools
import operator
import random
from collections import Counter

import numpy as np


def has_no_neighbor(user_consumed, item_consumed, item):
    for u in item_consumed[item]:
        if len(user_consumed[u]) > 1:
            return False
    return True


def bipartite_one_walk(user_consumed, item_consumed, item):
    user = random.choice(item_consumed[item])
    return random.choice(user_consumed[user])


def pairs_from_random_walk(
    start_nodes, user_consumed, item_consumed, num_walks, walk_length, focus_start
):
    items, items_pos = [], []
    for node in start_nodes:
        if has_no_neighbor(user_consumed, item_consumed, node):
            print(f"no neighbor to walk for item node {node}")
            items.append(node)
            items_pos.append(node)
        for _ in range(num_walks):
            cur_node = node
            for _ in range(walk_length):
                next_node = bipartite_one_walk(user_consumed, item_consumed, cur_node)
                if focus_start and next_node != node:
                    items.append(node)
                    items_pos.append(next_node)
                elif not focus_start and next_node != cur_node:
                    items.append(cur_node)
                    items_pos.append(next_node)
                cur_node = next_node
    return np.array(items), np.array(items_pos)


def bipartite_neighbors(
    nodes, user_consumed, item_consumed, num_neighbors, tolerance=5
):
    batch_neighbors, neighbor_lens = [], []
    for node in nodes:
        neighbors = []
        for _ in range(num_neighbors):
            n = bipartite_one_walk(user_consumed, item_consumed, node)
            if n == node or n in neighbors:
                success = False
                for _ in range(tolerance):
                    n = bipartite_one_walk(user_consumed, item_consumed, node)
                    if n != node and n not in neighbors:
                        success = True
                        break
                if not success:
                    for _ in range(tolerance):
                        n = bipartite_one_walk(user_consumed, item_consumed, node)
                        if n != node:
                            success = True
                            break
                if not success:
                    n = bipartite_one_walk(user_consumed, item_consumed, node)
                    # print(f"possible not enough neighbors for item {node}.")
            neighbors.append(n)
        batch_neighbors.extend(neighbors)
        neighbor_lens.append(len(neighbors))
    return batch_neighbors, compute_offsets(neighbor_lens)


def bipartite_neighbors_with_weights(
    nodes,
    user_consumed,
    item_consumed,
    num_neighbors,
    num_walks,
    walk_length,
    items=None,
    item_indices=None,
    items_pos=None,
    termination_prob=0.5,
):
    """Simulate PinSage-like random walk from a bipartite graph.

    The process will try to remove self item node during i2i training.
    However, if the item node has no neighbor, then itself will be returned.
    If the item node only has one neighbor, then that neighbor will be returned.
    Otherwise, next walk will be sampled randomly from neighbors.
    """

    def compute_weights(neighbors):
        if len(neighbors) == 1:
            return [neighbors[0]], [1.0]
        important_neighbors, weights = [], []
        for n, w in Counter(neighbors).most_common(num_neighbors):
            important_neighbors.append(n)
            weights.append(w)
        total_weights = sum(weights)
        importance_weights = [i / total_weights for i in weights]
        return important_neighbors, importance_weights

    batch_neighbors, batch_weights, neighbor_lens = [], [], []
    original_item_indices = [] if item_indices else None
    for i, node in enumerate(nodes):
        if has_no_neighbor(user_consumed, item_consumed, node):
            neighbors, weights = [node], [1.0]
        else:
            neighbors = []
            for _ in range(num_walks):
                walk = []
                for _ in range(walk_length):
                    if not walk:
                        # first walk
                        walk.append(
                            bipartite_one_walk(user_consumed, item_consumed, node)
                        )
                    elif random.random() >= termination_prob:
                        walk.append(
                            bipartite_one_walk(user_consumed, item_consumed, walk[-1])
                        )
                    else:
                        break
                assert len(walk) > 0
                neighbors.extend(walk)
            neighbors = remove_target_node(neighbors, node)
            if items is not None and item_indices is not None and items_pos is not None:
                index = item_indices[i]
                if items[index] == node:
                    neighbors = remove_target_node(neighbors, items_pos[index])
            neighbors, weights = compute_weights(neighbors)
        batch_neighbors.extend(neighbors)
        batch_weights.extend(weights)
        neighbor_lens.append(len(neighbors))
        if original_item_indices is not None:
            original_item_indices.extend([item_indices[i]] * len(neighbors))

    return (
        batch_neighbors,
        batch_weights,
        compute_offsets(neighbor_lens),
        original_item_indices,
    )


def remove_target_node(neighbors, node):
    num = neighbors.count(node)
    if num == 0:
        return neighbors
    elif num == len(neighbors):
        return [node]
    return [i for i in neighbors if i != node]


def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]
