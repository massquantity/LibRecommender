import subprocess
import time
from pathlib import Path

import requests

from libserving.serialization import embed2redis, save_embed, save_faiss_index
from tests.utils_path import SAVE_PATH, remove_path


def test_embed_serving(embed_model, redis_client):
    save_embed(SAVE_PATH, embed_model)
    embed2redis(SAVE_PATH)
    faiss_path = str(Path(__file__).parents[2] / "libserving" / "embed_model")
    save_faiss_index(faiss_path, embed_model, 40, 10)

    subprocess.run(["pkill", "sanic"])
    subprocess.run(
        "sanic libserving.sanic_serving.embed_deploy:app --no-access-logs --workers 2 &",
        shell=True,
    )
    time.sleep(1)  # wait for the server to start

    response = requests.post(
        "http://localhost:8000/embed/recommend", json={"user": 1, "n_rec": 1}
    )
    assert len(list(response.json().values())[0]) == 1
    response = requests.post(
        "http://localhost:8000/embed/recommend", json={"user": 33, "n_rec": 3}
    )
    assert len(list(response.json().values())[0]) == 3

    subprocess.run(["pkill", "sanic"])
    remove_path(faiss_path)
