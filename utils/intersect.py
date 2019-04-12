import numpy as np
import pickle
from .similarities import *


def get_intersect(dataset, sim_option="pearson", min_support=1, k=40, load=False, parallel=False):
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

    intersect_user_item_train = {}
    for u, i in zip(dataset.train_user_indices, dataset.train_item_indices):
        u_items = list(dataset.train_user[u].keys())
        sim_items = sim_whole[i]
        intersect_items, index_u, _ = np.intersect1d(
            u_items, sim_items, assume_unique=True, return_indices=True)
        intersect_user_item_train[(u, i)] = (intersect_items, index_u)

    return intersect_user_item_train

