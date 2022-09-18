import os
from typing import Union

from scipy import sparse

from libreco.algorithms import ItemCF, UserCF
from .common import (
    check_path_exists,
    save_id_mapping,
    save_model_name,
    save_to_json,
    save_user_consumed,
)


def save_knn(path: str, model: Union[UserCF, ItemCF], k: int):
    check_path_exists(path)
    save_model_name(path, model)
    save_id_mapping(path, model.data_info)
    save_user_consumed(path, model.data_info)
    save_sim_matrix(path, model.sim_matrix, k)


def save_sim_matrix(path: str, sim_matrix: sparse.csr_matrix, k: int):
    k_sims = dict()
    num = len(sim_matrix.indptr) - 1
    indices = sim_matrix.indices.tolist()
    indptr = sim_matrix.indptr.tolist()
    data = sim_matrix.data.tolist()
    for i in range(num):
        i_slice = slice(indptr[i], indptr[i + 1])
        sorted_sims = sorted(zip(indices[i_slice], data[i_slice]), key=lambda x: -x[1])
        k_sims[i] = sorted_sims[:k]
    sim_path = os.path.join(path, "sim.json")
    save_to_json(sim_path, k_sims)
