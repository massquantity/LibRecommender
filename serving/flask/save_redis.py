import json
import redis


r = redis.Redis(host="localhost", port=6379, decode_responses=True)


def similarity2redis(path, name):
    with open(path, "r") as f:
        similarities = f.read()
    r.set(name, similarities)


def vector2redis(path, name):
    with open(path, "r") as f:
        vector = json.load(f)
    vector_str = {k: str(v) for k, v in vector.items()}
    r.hmset(name, vector_str)


def user_consumed2redis(path, name):
    with open(path, "r") as f:
        user_consumed_str = f.read()
    r.set(name, user_consumed_str)


def data_info2redis(path):
    with open(path, "r") as f:
        data_info = json.load(f)
    for k, v in data_info.items():
        r.set(k, str(v))

