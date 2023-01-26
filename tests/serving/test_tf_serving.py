import os
import subprocess
import time

import pytest
import requests

from libreco.bases import TfBase
from libserving.serialization import save_tf, tf2redis
from tests.utils_path import SAVE_PATH


@pytest.mark.parametrize(
    "tf_model", ["pure", "feat-all", "feat-user", "feat-item"], indirect=True
)
def test_tf_serving(tf_model, redis_client):
    assert isinstance(tf_model, TfBase)
    save_tf(SAVE_PATH, tf_model, version=1)
    tf2redis(SAVE_PATH)

    subprocess.run(["pkill", "sanic"])
    subprocess.run("kill $(lsof -t -i:8501 -sTCP:LISTEN) 2> /dev/null", shell=True)
    time.sleep(0.1)
    subprocess.run(
        "sanic libserving.sanic_serving.tf_deploy:app --no-access-logs --workers 2 &",
        shell=True,
    )
    subprocess.run("python tests/serving/mock_tf_server.py &", shell=True)
    time.sleep(1)  # wait for the server to start

    response = requests.post(
        "http://localhost:8000/tf/recommend", json={"user": 1, "n_rec": 1}
    )
    assert len(list(response.json().values())[0]) == 1
    response = requests.post(
        "http://localhost:8000/tf/recommend", json={"user": 33, "n_rec": 3}
    )
    assert len(list(response.json().values())[0]) == 3

    subprocess.run(["pkill", "sanic"])
    subprocess.run("kill $(lsof -t -i:8501 -sTCP:LISTEN)", shell=True)
