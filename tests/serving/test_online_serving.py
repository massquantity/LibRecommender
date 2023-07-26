import subprocess

import pytest

from libserving.serialization import online2redis, save_online
from tests.utils_data import SAVE_PATH


@pytest.mark.parametrize(
    "online_model",
    ["pure", "user_feat", "separate", "multi_sparse", "item_feat", "all"],
    indirect=True,
)
def test_online_serving(online_model, session, close_server):
    save_online(SAVE_PATH, online_model, version=1)
    online2redis(SAVE_PATH)

    subprocess.run(
        "sanic libserving.sanic_serving.online_deploy:app --no-access-logs --single-process &",
        shell=True,
        check=True,
    )
    subprocess.run("python tests/serving/mock_tf_server.py &", shell=True, check=True)
    # time.sleep(2)  # wait for the server to start

    response = session.post(
        "http://localhost:8000/online/recommend",
        json={"user": 1, "n_rec": 1},
        timeout=0.5,
    )
    assert len(next(iter(response.json().values()))) == 1

    response = session.post(
        "http://localhost:8000/online/recommend",
        json={"user": "uuu", "n_rec": 3},
        timeout=0.5,
    )
    assert len(next(iter(response.json().values()))) == 3

    response = session.post(
        "http://localhost:8000/online/recommend",
        json={"user": 2, "n_rec": 3, "user_feats": {"sex": "male"}},
        timeout=0.5,
    )
    assert len(next(iter(response.json().values()))) == 3

    response = session.post(
        "http://localhost:8000/online/recommend",
        json={"user": 2, "n_rec": 3, "seq": [1, 2, 3, 10, 11, 11, 22, 1, 0, -1, 12, 1]},
        timeout=0.5,
    )
    assert len(next(iter(response.json().values()))) == 3

    response = session.post(
        "http://localhost:8000/online/recommend",
        json={
            "user": "uuu",
            "n_rec": 30000,
            "user_feats": {"sex": "bb", "age": 1000, "occupation": "ooo", "ggg": "eee"},
            "seq": [1, 2, 3, "??"],
        },
        timeout=0.5,
    )
    # noinspection PyUnresolvedReferences
    assert len(next(iter(response.json().values()))) == online_model.n_items

    response = session.post(
        "http://localhost:8000/online/recommend",
        json={
            "user": "uuu",
            "n_rec": 300,
            "item_feats": {"sex": "bb", "age": 1000, "occupation": "ooo", "ggg": "eee"},
            "item_seq": [1, 2, 3, "??"],
        },
        timeout=0.5,
    )
    assert "Invalid payload" in response.text
