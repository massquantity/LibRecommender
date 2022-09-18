import contextlib
import os

import redis
import ujson


@contextlib.contextmanager
def redis_connection(host: str, port: int, db: int):
    r = None
    try:
        r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        yield r
    except (redis.ConnectionError, redis.DataError):
        raise
    finally:
        if r:
            r.close()


def knn2redis(path: str, host: str = "localhost", port: int = 6379, db: int = 0):
    with redis_connection(host, port, db) as r:
        model_name2redis(path, r)
        id_mapping2redis(path, r)
        user_consumed2redis(path, r)
        sim2redis(path, r)


def embed2redis(path: str, host: str = "localhost", port: int = 6379, db: int = 0):
    with redis_connection(host, port, db) as r:
        model_name2redis(path, r)
        id_mapping2redis(path, r)
        user_consumed2redis(path, r)
        user_embed2redis(path, r)


def tf2redis(path: str, host: str = "localhost", port: int = 6379, db: int = 0):
    with redis_connection(host, port, db) as r:
        model_name2redis(path, r)
        id_mapping2redis(path, r)
        user_consumed2redis(path, r)
        features2redis(path, r)


def model_name2redis(path: str, r: redis.Redis):
    model_name_path = os.path.join(path, "model_name.json")
    with open(model_name_path) as f:
        m = ujson.load(f)
    r.set("model_name", m["model_name"])


def id_mapping2redis(path: str, r: redis.Redis):
    user2id_path = os.path.join(path, "user2id.json")
    id2item_path = os.path.join(path, "id2item.json")
    with open(user2id_path) as f1, open(id2item_path) as f2:
        user2id = ujson.load(f1)
        id2item = ujson.load(f2)
    r.hset("user2id", mapping=user2id)
    r.hset("id2item", mapping=id2item)


def user_consumed2redis(path: str, r: redis.Redis):
    user_consumed_path = os.path.join(path, "user_consumed.json")
    with open(user_consumed_path) as f:
        user_consumed = ujson.load(f)
    pipe = r.pipeline()
    for u, items in user_consumed.items():
        pipe.hset("user_consumed", u, ujson.dumps(items))
    pipe.execute()


def sim2redis(path: str, r: redis.Redis):
    sim_path = os.path.join(path, "sim.json")
    with open(sim_path) as f:
        sim = ujson.load(f)
    pipe = r.pipeline()
    for k, k_sims in sim.items():
        pipe.hset("k_sims", k, ujson.dumps(k_sims))
    pipe.execute()


def user_embed2redis(path: str, r: redis.Redis):
    embed_path = os.path.join(path, "user_embed.json")
    with open(embed_path) as f:
        user_embeds = ujson.load(f)
    pipe = r.pipeline()
    for u, embed in user_embeds.items():
        pipe.hset("user_embed", u, ujson.dumps(embed))
    pipe.execute()


def features2redis(path: str, r: redis.Redis):
    feature_path = os.path.join(path, "features.json")
    with open(feature_path) as f:
        feats = ujson.load(f)

    r.set("n_items", feats["n_items"])
    if "max_seq_len" in feats:
        r.set("max_seq_len", feats["max_seq_len"])

    if "user_sparse_col_index" in feats:
        r.hset("feature", "user_sparse", 1)
        r.set("user_sparse_col_index", ujson.dumps(feats["user_sparse_col_index"]))
        pipe = r.pipeline()
        for u, vals in enumerate(feats["user_sparse_values"]):
            pipe.hset("user_sparse_values", str(u), ujson.dumps(vals))
        pipe.execute()

    if "item_sparse_col_index" in feats:
        r.hset("feature", "item_sparse", 1)
        r.set("item_sparse_col_index", ujson.dumps(feats["item_sparse_col_index"]))
        pipe = r.pipeline()
        for vals in feats["item_sparse_values"]:
            pipe.rpush("item_sparse_values", ujson.dumps(vals))
        pipe.execute()

    if "user_dense_col_index" in feats:
        r.hset("feature", "user_dense", 1)
        r.set("user_dense_col_index", ujson.dumps(feats["user_dense_col_index"]))
        pipe = r.pipeline()
        for u, vals in enumerate(feats["user_dense_values"]):
            pipe.hset("user_dense_values", str(u), ujson.dumps(vals))
        pipe.execute()

    if "item_dense_col_index" in feats:
        r.hset("feature", "item_dense", 1)
        r.set("item_dense_col_index", ujson.dumps(feats["item_dense_col_index"]))
        pipe = r.pipeline()
        for vals in feats["item_dense_values"]:
            pipe.rpush("item_dense_values", ujson.dumps(vals))
        pipe.execute()
