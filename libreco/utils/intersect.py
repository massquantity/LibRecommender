import numpy as np
import os
import pickle
from .similarities import *
from .baseline_estimates import baseline_als
import tensorflow as tf


def get_intersect(dataset, sim_option="pearson", min_support=1, k=40, load=False, parallel=False):
    n = len(dataset.train_item)
    ids = list(dataset.train_item.keys())
    if load:
        with open(os.path.expanduser("~/Workspace/LibRecommender/sim.pkl"), "rb") as f:
            sim_matrix = pickle.load(f)
    elif parallel:
        sim_matrix = get_sim_parallel(dataset.train_item, sim_option,
                                      ids, ids, min_support=min_support)
    else:
        sim_matrix = get_sim(dataset.train_item, sim_option, n, ids, min_support=min_support)
    print("similarity matrix shape: ", sim_matrix.shape)

    sim_whole = {}
    for i in range(dataset.n_items):
        sim_whole[i] = np.argsort(sim_matrix[i])[::-1][:k]

#    intersect_user_item_train = {}
    intersect_items_all = []
    intersect_indices_all = []
    u_labels_all = []
#    base_neighbor_all = []
    for u, i in zip(dataset.train_user_indices, dataset.train_item_indices):
        u_items = list(dataset.train_user[u].keys())
        sim_items = sim_whole[i]
        intersect_items, index_u, _ = np.intersect1d(
            u_items, sim_items, assume_unique=True, return_indices=True)
    #    intersect_user_item_train[(u, i)] = (intersect_items, index_u)
        intersect_items_all.append(list(intersect_items))
        intersect_indices_all.append(list(index_u))
        u_labels_all.append(list(np.array(list(dataset.train_user[u].values()))[index_u]))
#        base_neighbor = dataset.global_mean + baseline_user[u] + baseline_item[intersect_items]
#        base_neighbor_all.append(list(base_neighbor))
#    return intersect_user_item_train
    return intersect_items_all, intersect_indices_all, u_labels_all


def get_intersect_tf(dataset, sim_option="pearson", min_support=1, k=40, load=False, parallel=False):
    n = len(dataset.train_item)
    ids = list(dataset.train_item.keys())
    if load:
        sim_matrix = pickle.load(open("test/sim_matrix.pkl", "rb"))
    elif parallel:
        sim_matrix = get_sim_parallel(dataset.train_item, sim_option,
                                      ids, ids, min_support=min_support)
    else:
        sim_matrix = get_sim(dataset.train_item, sim_option, n, ids, min_support=min_support)
    print("similarity matrix shape: ", sim_matrix.shape)

    sim_whole = {}
    for i in range(dataset.n_items):
        sim_whole[i] = np.argsort(sim_matrix[i])[::-1][:k]

    offset = 0
    sparse_dict = {'indices': [], 'values': [], 'dense_shape': (dataset.n_items * dataset.n_items, 1)}
    sparse_weight = {'indices': [], 'values': [], 'dense_shape': (dataset.n_items * dataset.n_items, 1)}
    item_container = set()
    for u, i, r in zip(dataset.train_user_indices, dataset.train_item_indices, dataset.train_ratings):
        u_items = list(dataset.train_user[u].keys())
        sim_items = sim_whole[i]
        intersect_items, index_u, _ = np.intersect1d(
            u_items, sim_items, assume_unique=True, return_indices=True)
        for user_index, item in zip(index_u, intersect_items):
            sparse_dict['indices'].append((u, 1))
            sparse_dict['values'].append(item + offset)

            u_rating = list(dataset.train_user[u].values())[user_index]
            bbu, bbi = baseline_als(dataset)
            base_neighbor = dataset.global_mean + bbu[u] + bbi[item]
            sparse_weight['indices'].append((u, 1))
            sparse_weight['values'] = u_rating - base_neighbor

        if i not in item_container:
            offset += dataset.n_items
            item_container.add(i)
    return tf.SparseTensor(**sparse_dict), tf.SparseTensor(**sparse_weight)


def get_sim(dataset, k=40):
    with open(os.path.expanduser("~/Workspace/LibRecommender/sim_item.pkl"), "rb") as f:  # item sim matrix
        sim_matrix = pickle.load(f)
    print("similarity matrix shape: ", sim_matrix.shape)

    sim_whole = []
    for i in range(dataset.n_items):
        sim_whole.append(np.argsort(sim_matrix[i])[::-1][:k].tolist())

    return sim_whole

