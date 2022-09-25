import os
from asyncio.events import AbstractEventLoop
from pathlib import Path
from typing import List

import faiss
import numpy as np
import redis.asyncio as redis
import ujson
from sanic import Sanic
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.request import Request
from sanic.response import HTTPResponse, json

from .common import Params, validate

app = Sanic("embed-serving")


@app.post("/embed/recommend")
@validate(model=Params)
async def embed_serving(request: Request, params: Params) -> HTTPResponse:
    r: redis.Redis = app.ctx.redis
    user = params.user
    n_rec = params.n_rec
    if not await r.hexists("user2id", user):
        raise SanicException(f"Invalid user {user} doesn't exist", status_code=400)

    logger.info(f"recommend {n_rec} items for user {user}")
    user_id = await r.hget("user2id", user)
    reco_list = await recommend_on_similar_embeds(user_id, n_rec, r)
    return json({f"Recommend result for user {user}": reco_list})


async def recommend_on_similar_embeds(
    user_id: str, n_rec: int, r: redis.Redis
) -> List[str]:
    u_consumed = set(ujson.loads(await r.hget("user_consumed", user_id)))
    candidate_num = n_rec + len(u_consumed)
    user_embed = ujson.loads(await r.hget("user_embed", user_id))
    if len(user_embed) != app.ctx.faiss_index.d:
        raise SanicException(
            "user_embed dimension != item_embed dimension, did u load the wrong faiss index?",
            status_code=500,
        )

    user_embed = np.array(user_embed, dtype=np.float32).reshape(1, -1)
    _, item_ids = app.ctx.faiss_index.search(user_embed, candidate_num)
    reco_list = []
    for i in item_ids.flatten().tolist():
        if i not in u_consumed:
            reco_list.append(await r.hget("id2item", i))
            if len(reco_list) == n_rec:
                break
    return reco_list


@app.before_server_start
async def redis_faiss_setup(app: Sanic, loop: AbstractEventLoop):
    app.ctx.redis = await redis.from_url("redis://localhost", decode_responses=True)
    app.ctx.faiss_index = faiss.read_index(find_index_path())


@app.after_server_stop
async def redis_close(app: Sanic):
    await app.ctx.redis.close()


def find_index_path():
    # par_dir = str(Path(os.path.realpath(__file__)).parent.parent)
    par_dir = str(Path(__file__).absolute().parent.parent)
    for dir_path, _, files in os.walk(par_dir):
        for file in files:
            if file.startswith("faiss_index"):
                return os.path.join(dir_path, file)
    raise SanicException(f"Failed to find faiss index in {par_dir}", status_code=500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, access_log=False)
