import subprocess
import time

import pytest
import requests

from libreco.bases import TfBase
from libserving.serialization import save_tf, tf2redis
from tests.utils_data import SAVE_PATH


@pytest.mark.parametrize(
    "tf_model", ["pure", "feat-all", "feat-user", "feat-item"], indirect=True
)
def test_tf_serving(tf_model, close_server):
    assert isinstance(tf_model, TfBase)
    save_tf(SAVE_PATH, tf_model, version=1)
    tf2redis(SAVE_PATH)

    subprocess.run(
        "sanic libserving.sanic_serving.tf_deploy:app --no-access-logs --workers 2 &",
        shell=True,
        check=True,
    )
    subprocess.run("python tests/serving/mock_tf_server.py &", shell=True, check=True)
    time.sleep(2)  # wait for the server to start

    response = requests.post(
        "http://localhost:8000/tf/recommend", json={"user": 1, "n_rec": 1}, timeout=1
    )
    assert len(next(iter(response.json().values()))) == 1
    response = requests.post(
        "http://localhost:8000/tf/recommend", json={"user": 33, "n_rec": 3}, timeout=1
    )
    assert len(next(iter(response.json().values()))) == 3
