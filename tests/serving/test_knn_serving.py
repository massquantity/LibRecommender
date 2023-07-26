import subprocess

import pytest

from libreco.bases import CfBase
from libserving.serialization import knn2redis, save_knn
from tests.utils_data import SAVE_PATH


@pytest.mark.parametrize("knn_model", ["UserCF", "ItemCF"], indirect=True)
def test_knn_serving(knn_model, session, close_server):
    assert isinstance(knn_model, CfBase)
    save_knn(SAVE_PATH, knn_model, k=10)
    knn2redis(SAVE_PATH)

    subprocess.run(
        "sanic libserving.sanic_serving.knn_deploy:app --no-access-logs --single-process &",
        shell=True,
        check=True,
    )
    # time.sleep(2)  # wait for the server to start

    response = session.post(
        "http://localhost:8000/knn/recommend", json={"user": 1, "n_rec": 1}, timeout=0.5
    )
    assert len(next(iter(response.json().values()))) == 1
    response = session.post(
        "http://localhost:8000/knn/recommend",
        json={"user": 33, "n_rec": 3},
        timeout=0.5,
    )
    assert len(next(iter(response.json().values()))) == 3
