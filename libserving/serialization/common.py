import os
import shutil
import sys

import ujson

from libreco.bases import Base
from libreco.data import DataInfo
from libreco.data.data_info import EmptyFeature
from libreco.utils.misc import colorize


def save_model_name(path: str, model: Base):
    model_name_path = os.path.join(path, "model_name.json")
    save_to_json(model_name_path, {"model_name": model.model_name})


def save_id_mapping(path: str, data_info: DataInfo):
    user2id_path = os.path.join(path, "user2id.json")
    user2id = {int(k): int(v) for k, v in data_info.user2id.items()}  # np.int64 -> int
    save_to_json(user2id_path, user2id)
    id2item_path = os.path.join(path, "id2item.json")
    id2item = {int(k): int(v) for k, v in data_info.id2item.items()}
    save_to_json(id2item_path, id2item)
    item2id_path = os.path.join(path, "item2id.json")
    item2id = {int(k): int(v) for k, v in data_info.item2id.items()}
    save_to_json(item2id_path, item2id)


def save_user_consumed(path: str, data_info: DataInfo):
    user_consumed_path = os.path.join(path, "user_consumed.json")
    # user_consumed = dict()
    # for u, items in data_info.user_consumed.items():
    #    user_consumed[int(u)] = items.tolist()
    save_to_json(user_consumed_path, data_info.user_consumed)


def save_to_json(path: str, data: dict):
    with open(path, "w") as f:
        ujson.dump(data, f, ensure_ascii=False)


def check_path_exists(path):
    assert isinstance(path, str) and path, f"invalid saving path: `{path}`"
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)


def save_features(path: str, data_info: DataInfo, model):
    feats = {"n_users": data_info.n_users, "n_items": data_info.n_items}
    if data_info.col_name_mapping:
        if data_info.user_sparse_col != EmptyFeature:
            _check_num_match(data_info.user_sparse_unique, data_info.n_users)
            feats["user_sparse_col_index"] = data_info.user_sparse_col.index
            feats["user_sparse_values"] = data_info.user_sparse_unique.tolist()
        if data_info.item_sparse_col != EmptyFeature:
            _check_num_match(data_info.item_sparse_unique, data_info.n_items)
            feats["item_sparse_col_index"] = data_info.item_sparse_col.index
            feats["item_sparse_values"] = data_info.item_sparse_unique.tolist()
        if data_info.user_dense_col != EmptyFeature:
            _check_num_match(data_info.user_dense_unique, data_info.n_users)
            feats["user_dense_col_index"] = data_info.user_dense_col.index
            feats["user_dense_values"] = data_info.user_dense_unique.tolist()
        if data_info.item_dense_col != EmptyFeature:
            _check_num_match(data_info.item_dense_unique, data_info.n_items)
            feats["item_dense_col_index"] = data_info.item_dense_col.index
            feats["item_dense_values"] = data_info.item_dense_unique.tolist()

    if hasattr(model, "max_seq_len"):
        feats["max_seq_len"] = model.max_seq_len
    feature_path = os.path.join(path, "features.json")
    save_to_json(feature_path, feats)


# include oov
def _check_num_match(v, num):
    assert len(v) == num + 1, f"feature sizes don't match, got {len(v)} and {num + 1}"


def check_model_exists(export_path: str):  # pragma: no cover
    answered = False
    while not answered:
        print_str = (
            f"Could not export model because '{export_path}' "
            f"already exists, would you like to remove it? [Y/n]"
        )
        print(f"{colorize(print_str, 'red')}", end="")
        choice = input().lower()
        if choice in ["yes", "y"]:
            shutil.rmtree(export_path)
            answered = True
        elif choice in ["no", "n"]:
            print(f"{colorize('refused to remove, then exit...', 'red')}")
            sys.exit(0)
