import os

import ujson

from libreco.bases import Base
from libreco.data import DataInfo


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
