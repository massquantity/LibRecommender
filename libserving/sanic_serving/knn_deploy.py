from collections import defaultdict
from typing import List

import redis.asyncio as redis
import ujson
from sanic import Sanic
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.request import Request
from sanic.response import HTTPResponse, json

from .common import Params, validate

app = Sanic("knn-serving")


@app.post("/knn/recommend")
@validate(model=Params)
async def knn_serving(request: Request, params: Params) -> HTTPResponse:
    r: redis.Redis = app.ctx.redis
    user = params.user
    n_rec = params.n_rec
    if not await r.hexists("user2id", user):
        raise SanicException(f"Invalid user {user} doesn't exist", status_code=400)

    logger.info(f"recommend {n_rec} items for user {user}")
    user_id = await r.hget("user2id", user)
    model_name = await r.get("model_name")
    if model_name == "UserCF":
        reco_list = await recommend_on_user_similarities(user_id, n_rec, r)
    elif model_name == "ItemCF":
        reco_list = await recommend_on_item_similarities(user_id, n_rec, r)
    else:
        raise SanicException(f"Unknown knn model: {model_name}", status_code=500)
    return json({f"Recommend result for user {user}": reco_list})


async def recommend_on_user_similarities(
    user_id: str, n_rec: int, r: redis.Redis
) -> List[str]:
    u_consumed = set(ujson.loads(await r.hget("user_consumed", user_id)))
    k_sim_users = ujson.loads(await r.hget("k_sims", user_id))
    result = defaultdict(lambda: 0.0)
    for v, sim in k_sim_users:
        v_consumed = ujson.loads(await r.hget("user_consumed", v))
        for i in v_consumed:
            if i in u_consumed:
                continue
            result[i] += sim
    ranked_items = [
        i[0] for i in sorted(result.items(), key=lambda x: x[1], reverse=True)
    ]
    return [await r.hget("id2item", i) for i in ranked_items[:n_rec]]


async def recommend_on_item_similarities(
    user_id: str, n_rec: int, r: redis.Redis
) -> List[str]:
    u_consumed = set(ujson.loads(await r.hget("user_consumed", user_id)))
    result = defaultdict(lambda: 0.0)
    for i in u_consumed:
        k_sim_items = ujson.loads(await r.hget("k_sims", str(i)))
        for j, sim in k_sim_items:
            if j in u_consumed:
                continue
            result[j] += sim
    ranked_items = [
        i[0] for i in sorted(result.items(), key=lambda x: x[1], reverse=True)
    ]
    return [await r.hget("id2item", i) for i in ranked_items[:n_rec]]


@app.before_server_start
async def redis_setup(app: Sanic):
    app.ctx.redis = await redis.from_url("redis://localhost", decode_responses=True)


@app.after_server_stop
async def redis_close(app: Sanic):
    await app.ctx.redis.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, access_log=False)
