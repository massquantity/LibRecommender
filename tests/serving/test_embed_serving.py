import subprocess
import time
from pathlib import Path

import requests

from libserving.serialization import embed2redis, save_embed, save_faiss_index
from tests.utils_data import SAVE_PATH, remove_path


def test_embed_serving(embed_model, close_server):
    save_embed(SAVE_PATH, embed_model)
    embed2redis(SAVE_PATH)
    faiss_path = str(Path(__file__).parents[2] / "libserving" / "embed_model")
    save_faiss_index(faiss_path, embed_model, 40, 10)

    subprocess.run(
        "sanic libserving.sanic_serving.embed_deploy:app --no-access-logs --workers 2 &",
        shell=True,
        check=True,
    )
    time.sleep(2)  # wait for the server to start

    response = requests.post(
        "http://localhost:8000/embed/recommend", json={"user": 1, "n_rec": 1}, timeout=1
    )
    assert len(next(iter(response.json().values()))) == 1
    response = requests.post(
        "http://localhost:8000/embed/recommend",
        json={"user": 33, "n_rec": 3},
        timeout=1,
    )
    assert len(next(iter(response.json().values()))) == 3
    remove_path(faiss_path)
