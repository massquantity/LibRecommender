import subprocess
import time

import pytest
import requests

from libreco.bases import CfBase
from libserving.serialization import knn2redis, save_knn
from tests.utils_path import SAVE_PATH


@pytest.mark.parametrize("knn_model", ["UserCF", "ItemCF"], indirect=True)
def test_knn_serving(knn_model):
    assert isinstance(knn_model, CfBase)
    save_knn(SAVE_PATH, knn_model, k=10)
    knn2redis(SAVE_PATH)

    subprocess.run(["pkill", "sanic"], check=False)
    subprocess.run(
        "sanic libserving.sanic_serving.knn_deploy:app --no-access-logs --workers 2 &",
        shell=True,
        check=True,
    )
    time.sleep(3)  # wait for the server to start

    response = requests.post(
        "http://localhost:8000/knn/recommend", json={"user": 1, "n_rec": 1}, timeout=1
    )
    assert len(list(response.json().values())[0]) == 1
    response = requests.post(
        "http://localhost:8000/knn/recommend", json={"user": 33, "n_rec": 3}, timeout=1
    )
    assert len(list(response.json().values())[0]) == 3

    subprocess.run(["pkill", "sanic"], check=False)
