import os
from typing import Any, Dict, List, Optional, Tuple, Union

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

USER_EMBED_MODELS = ("RNN4Rec", "Caser", "WaveNet", "YouTubeRetrieval")
SEPARATE_FEAT_MODELS = "TwoTower"
CROSS_FEAT_MODELS = ("WideDeep", "FM", "DeepFM", "AutoInt", "YouTubeRanking", "DIN")
SPARSE_SEQ_MODELS = "YouTubeRetrieval"
SEQ_MODELS = ("RNN4Rec", "Caser", "WaveNet", "YouTubeRanking", "DIN")

app = Sanic("online-serving")


@app.post("/online/recommend")
@validate(model=Params)
async def online_serving(request: Request, params: Params) -> HTTPResponse:
    r: redis.Redis = app.ctx.redis
    user = params.user
    n_rec = params.n_rec
    user_feats = params.user_feats
    seq = params.seq
    logger.info(f"recommend {n_rec} items for user {user}")
    if user_feats:
        logger.info(f"user features: `{user_feats}`")
    if seq:
        logger.info(f"seq: `{seq}`")

    if await r.hexists("user2id", user):
        user_id = await r.hget("user2id", user)
    else:
        user_id = await r.get("n_users")
    reco_list = await recommend_online_features(user_id, n_rec, user_feats, seq, r)
    return json({f"Recommend result for user {user}": reco_list})


async def recommend_online_features(
    user_id: str,
    n_rec: int,
    user_feats: Optional[Dict[str, Union[str, int, float]]],
    user_seq: Optional[List[str]],
    r: redis.Redis,
) -> List[str]:
    model_name = await r.get("model_name")
    n_items = int(await r.get("n_items"))
    if await r.hexists("user_consumed", user_id):
        u_consumed = ujson.loads(await r.hget("user_consumed", user_id))
        candidate_num = min(n_rec + len(u_consumed), n_items)
    else:
        u_consumed = []
        candidate_num = min(n_rec, n_items)

    if model_name in USER_EMBED_MODELS:
        features = await build_user_embed_features(
            model_name, user_id, n_items, user_feats, r
        )
    elif model_name == SEPARATE_FEAT_MODELS:
        features = await build_separate_features(
            model_name, user_id, n_items, user_feats, r
        )
    else:
        features = await build_cross_features(
            model_name, user_id, n_items, user_feats, r
        )

    if model_name in SEQ_MODELS or model_name == SPARSE_SEQ_MODELS:
        features.update(await get_seq(model_name, user_seq, u_consumed, n_items, r))

    features.update({"k": candidate_num})
    ranked_items = await request_tf_serving(model_name, features)
    return await convert_items(ranked_items, n_rec, u_consumed, r)


async def build_user_embed_features(
    model_name: str,
    user_id: str,
    n_items: int,
    user_feats: Optional[Dict[str, Any]],
    r: redis.Redis,
) -> Dict[str, List[List[Union[int, float]]]]:
    features = dict()  # no `item_indices` in UserEmbedModels
    if model_name not in ("RNN4Rec", "YouTubeRetrieval"):
        features.update({"user_indices": [int(user_id)]})
    if model_name == "YouTubeRetrieval":
        user_sparse_feats, user_dense_feats = await split_user_feats(user_feats, r)
        user_sparse_vals = await get_user_feats(
            model_name, user_id, n_items, "user_sparse_values", user_sparse_feats, r
        )
        user_dense_vals = await get_user_feats(
            model_name, user_id, n_items, "user_dense_values", user_dense_feats, r
        )
        if user_sparse_vals:
            features.update({"user_sparse_indices": user_sparse_vals})
        if user_dense_vals:
            features.update({"user_dense_values": user_dense_vals})

    return features


async def build_separate_features(
    model_name: str,
    user_id: str,
    n_items: int,
    user_feats: Optional[Dict[str, Any]],
    r: redis.Redis,
) -> Dict[str, List[List[Union[int, float]]]]:
    features = {
        "user_indices": [int(user_id)],
        "item_indices": list(range(n_items)),
    }
    user_sparse_feats, user_dense_feats = await split_user_feats(user_feats, r)
    user_sparse_vals = await get_user_feats(
        model_name, user_id, n_items, "user_sparse_values", user_sparse_feats, r
    )
    user_dense_vals = await get_user_feats(
        model_name, user_id, n_items, "user_dense_values", user_dense_feats, r
    )
    if user_sparse_vals:
        features.update({"user_sparse_indices": user_sparse_vals})
    if user_dense_vals:
        features.update({"user_dense_values": user_dense_vals})

    item_sparse_vals = await get_item_feats(n_items, "item_sparse_values", r)
    item_dense_vals = await get_item_feats(n_items, "item_dense_values", r)
    if item_sparse_vals:
        features.update({"item_sparse_indices": item_sparse_vals})
    if item_dense_vals:
        features.update({"item_dense_values": item_dense_vals})

    return features


async def build_cross_features(
    model_name: str,
    user_id: str,
    n_items: int,
    user_feats: Optional[Dict[str, Any]],
    r: redis.Redis,
) -> Dict[str, List[List[Union[int, float]]]]:
    features = {
        "user_indices": np.full(n_items, int(user_id)).tolist(),
        "item_indices": list(range(n_items)),
    }

    sparse_vals = dense_vals = None
    has_user_sparse = await r.exists("user_sparse_col_index")
    has_item_sparse = await r.exists("item_sparse_col_index")
    has_user_dense = await r.exists("user_dense_col_index")
    has_item_dense = await r.exists("item_dense_col_index")
    user_sparse_feats, user_dense_feats = await split_user_feats(user_feats, r)

    if has_user_sparse and has_item_sparse:
        sparse_vals = await get_cross_feats(
            model_name,
            "user_sparse_col_index",
            "item_sparse_col_index",
            "user_sparse_values",
            "item_sparse_values",
            user_id,
            n_items,
            user_sparse_feats,
            r,
        )
    elif has_user_sparse:
        sparse_vals = await get_user_feats(
            model_name, user_id, n_items, "user_sparse_values", user_sparse_feats, r
        )
    elif has_item_sparse:
        sparse_vals = await get_item_feats(n_items, "item_sparse_values", r)

    if has_user_dense and has_item_dense:
        dense_vals = await get_cross_feats(
            model_name,
            "user_dense_col_index",
            "item_dense_col_index",
            "user_dense_values",
            "item_dense_values",
            user_id,
            n_items,
            user_dense_feats,
            r,
        )
    elif has_user_dense:
        dense_vals = await get_user_feats(
            model_name, user_id, n_items, "user_dense_values", user_dense_feats, r
        )
    elif has_item_dense:
        dense_vals = await get_item_feats(n_items, "item_dense_values", r)

    if sparse_vals:
        features.update({"sparse_indices": sparse_vals})
    if dense_vals:
        features.update({"dense_values": dense_vals})

    return features


async def split_user_feats(
    user_feats: Optional[Dict[str, Any]], r: redis.Redis
) -> Tuple[Optional[dict], Optional[dict]]:
    user_sparse_feats, user_dense_feats = dict(), dict()
    if user_feats:
        for col, val in user_feats.items():
            if await r.hexists("user_sparse_fields", col):
                user_sparse_feats[col] = val
            elif await r.hexists("user_dense_fields", col):
                if not isinstance(val, (int, float)):
                    logger.warning(f"Got not a number dense val `{val}`.")
                user_dense_feats[col] = val
            else:
                logger.warning(f"Unknown feature `{col}`.")

    return user_sparse_feats or None, user_dense_feats or None


async def update_user_sparse_feats(
    user_sparse_vals: List[int],
    user_sparse_feats: Dict[str, int],
    r: redis.Redis,
) -> List[List[int]]:
    for col, val in user_sparse_feats.items():
        # col must exist in `user_sparse_fields`, already checked in `split_user_feats`
        field_index = int(await r.hget("user_sparse_fields", col))
        mapping_key = f"user_sparse_idx_mapping__{col}"
        if await r.hexists(mapping_key, val):
            feat_index = int(await r.hget(mapping_key, val))
            user_sparse_vals[field_index] = feat_index
        else:
            logger.warning(f"Unknown value `{val}` in sparse feature `{col}`.")
    return user_sparse_vals


async def update_user_dense_feats(
    user_dense_vals: List[Union[float, int]],
    user_dense_feats: Dict[str, Union[float, int]],
    r: redis.Redis,
) -> List[List[float]]:
    for col, val in user_dense_feats.items():
        field_index = int(await r.hget("user_dense_fields", col))
        # if not isinstance(val, (int, float)):
        #    logger.warning(f"Possible invalid val `{val}`: `{type(val)}` in dense feature `{col}`")
        type_fn = type(user_dense_vals[0])
        user_dense_vals[field_index] = type_fn(val)
    return user_dense_vals


async def get_index_from_redis(index_name: str, r: redis.Redis) -> Optional[List[int]]:
    if not await r.exists(index_name):
        return
    return ujson.loads(await r.get(index_name))


async def get_value_from_redis(
    value_name: str,
    r: redis.Redis,
    user_id: Optional[str] = None,
) -> Optional[List[Union[int, float]]]:
    if not await r.exists(value_name):
        return
    if user_id:
        return ujson.loads(await r.hget(value_name, user_id))
    else:
        all_values = await r.lrange(value_name, 0, -1)
        return [ujson.loads(v) for v in all_values]


async def get_user_feats(
    model_name: str,
    user_id: str,
    n_items: int,
    value_name: str,
    user_feats: Optional[Dict[str, Any]],
    r: redis.Redis,
) -> Optional[List[List[Union[int, float]]]]:
    user_vals = await get_value_from_redis(value_name, r, user_id)
    if user_vals and user_feats:
        if "sparse" in value_name:
            user_vals = await update_user_sparse_feats(user_vals, user_feats, r)
        elif "dense" in value_name:
            user_vals = await update_user_dense_feats(user_vals, user_feats, r)
    if user_vals:
        if model_name in CROSS_FEAT_MODELS:
            user_vals = np.tile(user_vals, (n_items, 1)).tolist()
        else:
            user_vals = [user_vals]
    return user_vals


async def get_item_feats(
    n_items: int, value_name: str, r: redis.Redis
) -> Optional[List[List[Union[int, float]]]]:
    item_vals = await get_value_from_redis(value_name, r)
    if item_vals:
        item_vals = item_vals[:n_items]
    return item_vals


async def get_cross_feats(
    model_name: str,
    user_index_name: str,
    item_index_name: str,
    user_value_name: str,
    item_value_name: str,
    user_id: str,
    n_items: int,
    user_feats: Optional[Dict[str, Any]],
    r: redis.Redis,
) -> List[List[Union[int, float]]]:
    user_col_index = await get_index_from_redis(user_index_name, r)
    item_col_index = await get_index_from_redis(item_index_name, r)
    user_values = np.array(
        await get_user_feats(
            model_name, user_id, n_items, user_value_name, user_feats, r
        )
    )
    item_values = np.array(await get_item_feats(n_items, item_value_name, r))
    dim = len(user_col_index) + len(item_col_index)
    features = np.empty((n_items, dim), dtype=item_values.dtype)
    for i, col_idx in enumerate(user_col_index):
        features[:, col_idx] = user_values[:, i]
    for i, col_idx in enumerate(item_col_index):
        features[:, col_idx] = item_values[:, i]
    return features.tolist()


async def get_seq(
    model_name: str,
    user_seq: Optional[List[str]],
    user_consumed: List[int],
    n_items: int,
    r: redis.Redis,
) -> Dict[str, Any]:
    if not await r.exists("max_seq_len"):
        raise SanicException(
            f"Missing `max_seq_len` attribute in {model_name}", status_code=500
        )

    if user_seq:
        item_id_seq = []
        for i in user_seq:
            if await r.hexists("item2id", i):
                item_id_seq.append(int(await r.hget("item2id", i)))
            else:
                item_id_seq.append(n_items)
    else:
        item_id_seq = user_consumed

    max_seq_len = int(await r.get("max_seq_len"))
    seq_len = min(max_seq_len, len(item_id_seq)) if item_id_seq else max_seq_len

    if model_name == SPARSE_SEQ_MODELS:
        # -1 will be pruned in `tf.nn.safe_embedding_lookup_sparse`
        if not item_id_seq:
            sparse_seq_indices = np.zeros((1, 2), dtype=np.int64)
            sparse_seq_values = np.array([-1], dtype=np.int64)
        else:
            sparse_seq_indices = np.zeros((seq_len, 2), dtype=np.int64)
            seq = [i if i < n_items else -1 for i in item_id_seq[-seq_len:]]
            sparse_seq_values = np.array(seq, dtype=np.int64)
        return {
            "item_interaction_indices": sparse_seq_indices.tolist(),
            "item_interaction_values": sparse_seq_values.tolist(),
            "modified_batch_size": 1,
        }
    else:
        seq = np.full(max_seq_len, n_items, dtype=np.int32)
        if item_id_seq:
            seq[:seq_len] = item_id_seq[-seq_len:]

        if model_name in CROSS_FEAT_MODELS:
            seq = np.tile(seq, (n_items, 1)).tolist()
            seq_len = [seq_len] * n_items
        else:
            seq = [seq.tolist()]
            seq_len = [seq_len]

        return {
            "user_interacted_seq": seq,
            "user_interacted_len": seq_len,
        }


async def request_tf_serving(
    model_name: str, features: Dict[str, List[Any]]
) -> List[int]:
    # for k, v in features.items():
    #    logger.warning(f"{k}: {np.array(v).shape}")

    host = os.getenv("TF_SERVING_HOST", "localhost")
    url = f"http://{host}:8501/v1/models/{model_name.lower()}:predict"
    data = {"signature_name": "topk", "inputs": features}  # signature_name: topk
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            if resp.status != 200:
                raise SanicException(
                    f"Error when requesting TensorFlow Serving model: {resp.text}",
                    status_code=500,
                )
            result = await resp.json()

    # logger.warning(result["outputs"])
    return result["outputs"]


async def convert_items(
    ranked_items: List[int],
    n_rec: int,
    u_consumed: List[int],
    r: redis.Redis,
) -> List[str]:
    consumed_set = set(u_consumed) if u_consumed else None
    reco_list = []
    for i in ranked_items:
        if consumed_set and i in consumed_set:
            continue
        reco_list.append(await r.hget("id2item", str(i)))
        if len(reco_list) == n_rec:
            break
    return reco_list


@app.before_server_start
async def redis_setup(app: Sanic):
    host = os.getenv("REDIS_HOST", "localhost")
    app.ctx.redis = await redis.from_url(f"redis://{host}", decode_responses=True)


@app.after_server_stop
async def redis_close(app: Sanic):
    await app.ctx.redis.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, access_log=False)
