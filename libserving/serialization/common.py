import json
import os

from libreco.data import DataInfo


def save_id_mapping(path: str, data_info: DataInfo):
    user2id_path = os.path.join(path, "user2id.json")
    user2id = {int(k): int(v) for k, v in data_info.user2id.items()}
    save_to_json(user2id_path, user2id)
    id2item_path = os.path.join(path, "id2item.json")
    id2item = {int(k): int(v) for k, v in data_info.id2item.items()}
    save_to_json(id2item_path, id2item)


def save_user_consumed(path: str, data_info: DataInfo):
    user_consumed_path = os.path.join(path, "user_consumed.json")
    user_consumed = dict()
    for u, items in data_info.user_consumed.items():
        user_consumed[int(u)] = items.tolist()
    save_to_json(user_consumed_path, user_consumed)


def save_to_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def check_path_exists(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)
