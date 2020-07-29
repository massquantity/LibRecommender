import json
import os
import redis


r = redis.Redis(host="localhost", port=6379, decode_responses=True)


def sim2redis(path, name="k_sims"):
    sim_path = os.path.join(path, "sim.json")
    with open(sim_path, "r") as f:
        similarities = f.read()
    r.set(name, similarities)


def vector2redis(path, name=("user_vector", "item_vector")):
    u_vec_path = os.path.join(path, "user_vector.json")
    with open(u_vec_path, "r") as f1:
        user_vector = json.load(f1)
    user_vector_str = {k: str(v) for k, v in user_vector.items()}
    r.hmset(name[0], user_vector_str)

    i_vec_path = os.path.join(path, "item_vector.json")
    with open(i_vec_path, "r") as f2:
        item_vector = json.load(f2)
    item_vector_str = {k: str(v) for k, v in item_vector.items()}
    r.hmset(name[1], item_vector_str)


def user_consumed2redis(path, name="user_consumed"):
    uc_path = os.path.join(path, "user_consumed.json")
    with open(uc_path, "r") as f:
        user_consumed_str = f.read()
    r.set(name, user_consumed_str)


def data_info2redis(path, name="data_info"):
    df_path = os.path.join(path, "data_info.json")
    with open(df_path, "r") as f:
        data_info = json.load(f)
    for k, v in data_info.items():
        r.set(k, str(v))


def seq2redis(path, name="user_last_interacted"):
    u_interacted_path = os.path.join(path, "user_last_interacted.json")
    with open(u_interacted_path, "r") as f:
        user_interacted = json.load(f)
    user_interacted_str = {k: str(v) for k, v in user_interacted.items()}
    r.hmset(name, user_interacted_str)

