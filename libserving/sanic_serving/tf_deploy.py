import os
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import redis.asyncio as redis
import ujson
from sanic import Sanic
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.request import Request
from sanic.response import HTTPResponse, json

from .common import Params, validate

app = Sanic("tf-serving")


@app.post("/tf/recommend")
@validate(model=Params)
async def tf_serving(request: Request, params: Params) -> HTTPResponse:
    r: redis.Redis = app.ctx.redis
    user = params.user
    n_rec = params.n_rec
    if not await r.hexists("user2id", user):
        raise SanicException(f"Invalid user {user} doesn't exist", status_code=400)

    logger.info(f"recommend {n_rec} items for user {user}")
    user_id = await r.hget("user2id", user)
    reco_list = await recommend_on_features(user_id, n_rec, r)
    return json({f"Recommend result for user {user}": reco_list})


async def recommend_on_features(user_id: str, n_rec: int, r: redis.Redis) -> List[str]:
    n_items = int(await r.get("n_items"))
    u_consumed = ujson.loads(await r.hget("user_consumed", user_id))
    candidate_num = n_rec + len(u_consumed)
    features = {
        "user_indices": np.full(n_items, int(user_id)).tolist(),
        "item_indices": list(range(n_items)),
    }

    user_sparse_col_index, user_sparse_values = await get_one_from_redis(
        "user_sparse_col_index", "user_sparse_values", user_id, r
    )
    item_sparse_col_index, item_sparse_values = await get_all_from_redis(
        "item_sparse_col_index", "item_sparse_values", r
    )
    if user_sparse_col_index or item_sparse_col_index:
        features.update(
            await build_features(
                user_sparse_col_index,
                item_sparse_col_index,
                user_sparse_values,
                item_sparse_values,
                n_items,
                "int",
                "sparse_indices",
            )
        )

    user_dense_col_index, user_dense_values = await get_one_from_redis(
        "user_dense_col_index", "user_dense_values", user_id, r
    )
    item_dense_col_index, item_dense_values = await get_all_from_redis(
        "item_dense_col_index", "item_dense_values", r
    )
    if user_dense_col_index or item_dense_col_index:
        features.update(
            await build_features(
                user_dense_col_index,
                item_dense_col_index,
                user_dense_values,
                item_dense_values,
                n_items,
                "float",
                "dense_values",
            )
        )

    model_name = await r.get("model_name")
    if model_name in ("YouTubeRanking", "DIN"):
        features.update(await get_last_interaction(model_name, u_consumed, n_items, r))

    scores = await request_tf_serving(features, model_name)
    return await rank_items_by_score(scores, n_rec, candidate_num, u_consumed, r)


async def get_one_from_redis(
    index_name: str,
    value_name: str,
    id_: str,
    r: redis.Redis,
) -> Optional[Tuple[List[int], List[Any]]]:
    if await r.exists(index_name):
        index = ujson.loads(await r.get(index_name))
        values = ujson.loads(await r.hget(value_name, id_))
    else:
        index = values = None
    return index, values


async def get_all_from_redis(
    index_name: str, value_name: str, r: redis.Redis
) -> Optional[Tuple[List[int], List[Any]]]:
    if await r.exists(index_name):
        index = ujson.loads(await r.get(index_name))
        all_values = await r.lrange(value_name, 0, -1)
        values = [ujson.loads(v) for v in all_values]
    else:
        index = values = None
    return index, values


async def build_features(
    user_col_index: List[int],
    item_col_index: List[int],
    user_values: List[Any],
    item_values: List[List[Any]],
    n_items: int,
    type_: str,
    feature_name: str,
) -> Dict[str, List[List[Any]]]:
    dtype = np.int32 if type_.startswith("int") else np.float32
    if user_col_index and item_col_index:
        dim = len(user_col_index) + len(item_col_index)
        features = np.empty((n_items, dim), dtype=dtype)
        for item_id in range(n_items):
            for i, v in zip(user_col_index, user_values):
                features[item_id, i] = v
            for i, v in zip(item_col_index, item_values[item_id]):
                features[item_id, i] = v
        return {feature_name: features.tolist()}
    elif user_col_index:
        features = np.empty(len(user_col_index), dtype=dtype)
        for i, v in zip(user_col_index, user_values):
            features[i] = v
        features = features.tolist()
        return {feature_name: [features] * n_items}
    else:
        features = np.empty((n_items, len(item_col_index)), dtype=dtype)
        for item_id in range(n_items):
            for i, v in zip(item_col_index, item_values[item_id]):
                features[item_id, i] = v
        return {feature_name: features.tolist()}


async def get_last_interaction(
    model_name: str, user_consumed: List[int], n_items: int, r: redis.Redis
) -> Dict[str, Any]:
    if not await r.exists("max_seq_len"):
        raise SanicException(
            f"Missing `max_seq_len` attribute in {model_name}", status_code=500
        )

    num_consumed = len(user_consumed)
    max_seq_len = int(await r.get("max_seq_len"))
    if num_consumed >= max_seq_len:
        u_last_interacted = user_consumed[-max_seq_len:]
        u_interacted_len = max_seq_len
    else:
        u_last_interacted = np.full(max_seq_len, n_items, dtype=np.int32)
        u_last_interacted[-num_consumed:] = user_consumed
        u_last_interacted = u_last_interacted.tolist()
        u_interacted_len = num_consumed
    return {
        "user_interacted_seq": [u_last_interacted] * n_items,
        "user_interacted_len": [u_interacted_len] * n_items,
    }


async def request_tf_serving(
    features: Dict[str, List[Any]], model_name: str
) -> List[float]:
    host = os.getenv("TF_SERVING_HOST", "localhost")
    url = f"http://{host}:8501/v1/models/{model_name.lower()}:predict"
    data = {"signature_name": "predict", "inputs": features}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            if resp.status != 200:
                raise SanicException(
                    f"Error when requesting TensorFlow Serving model: {resp.text}",
                    status_code=500,
                )
            result = await resp.json()
    return result["outputs"]


async def rank_items_by_score(
    scores: List[float],
    n_rec: int,
    candidate_num: int,
    u_consumed: List[int],
    r: redis.Redis,
) -> List[str]:
    scores = np.array(scores)
    ids = np.argpartition(scores, -candidate_num)[-candidate_num:]
    rank_items = sorted(zip(ids, scores[ids]), key=lambda x: x[1], reverse=True)
    consumed_set = set(u_consumed)
    reco_list = []
    for i, _ in rank_items:
        if i in consumed_set:
            continue
        reco_list.append(await r.hget("id2item", str(i)))
        if len(reco_list) == n_rec:
            break
    return reco_list


@app.before_server_start
async def redis_setup(app: Sanic):
    host = os.getenv("REDIS_HOST", "localhost")
    app.ctx.redis = await redis.from_url(f"redis://{host}", decode_responses=True)
    app.ctx.user_sparse = bool(await app.ctx.redis.hexists("feature", "user_sparse"))
    app.ctx.item_sparse = bool(await app.ctx.redis.hexists("feature", "item_sparse"))
    app.ctx.user_dense = bool(await app.ctx.redis.hexists("feature", "user_dense"))
    app.ctx.item_dense = bool(await app.ctx.redis.hexists("feature", "item_dense"))


@app.after_server_stop
async def redis_close(app: Sanic):
    await app.ctx.redis.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, access_log=False)
