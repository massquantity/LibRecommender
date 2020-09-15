from itertools import islice
import json
from flask import Flask, jsonify, request
import numpy as np
import redis
import requests
from serving.flask import colorize


app = Flask(__name__)

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
user_consumed = r.get("user_consumed")
user_consumed = json.loads(user_consumed)

sparse_col_reindex = json.loads(r.get("sparse_col_reindex"))
user_sparse_unique = json.loads(r.get("user_sparse_unique"))
item_sparse_unique = json.loads(r.get("item_sparse_unique"))
user_dense_unique = json.loads(r.get("user_dense_unique"))
n_items = len(item_sparse_unique)


@app.route("/<algo>/recommend", methods=['POST'])
def api_call(algo):
    test_json = request.get_json(force=True)
    test_json_str = f"test_json: {test_json}"
    print(f"{colorize(test_json_str, 'magenta', bold=True)}")

    try:
        test_data = json.loads(test_json) if isinstance(
            test_json, str) else test_json
        user = test_data["user"]
        n_rec = test_data["n_rec"]
    except json.JSONDecodeError:
        print("Could not parse json file...")
        raise
    except KeyError:
        return bad_request()

    reco_list = recommend(user, n_rec, algo)
    response = jsonify({f'recommend list for user ({user})': reco_list})
    return response


def recommend(user, n_rec, algo):
    u_consumed = set(user_consumed[user])
    count = n_rec + len(u_consumed)

    (user_indices,
     item_indices,
     sparse_indices,
     dense_values) = get_recommend_indices_and_values(user)
    features = []
    for ui, ii, si, dv in zip(
            user_indices, item_indices, sparse_indices, dense_values):
        features.append(
            {"user_indices": ui.tolist(), "item_indices": ii.tolist(),
             "sparse_indices": si.tolist(), "dense_values": dv.tolist()}
        )

    if algo.lower() in ("youtuberanking", "din"):
        u_last_interacted, u_interacted_len = get_user_last_interacted(user)
        for i, (uli, ul) in enumerate(zip(u_last_interacted, u_interacted_len)):
            features[i].update(
                {"user_interacted_seq": uli.tolist(),
                 "user_interacted_len": ul.tolist()}
            )

    data = {"signature_name": "predict", "instances": features}
    response = requests.post(
        f"http://localhost:8501/v1/models/{algo}:predict",
        data=json.dumps(data)
    )

#    recos = json.loads(response.text)["predictions"]
    recos = np.asarray(response.json()["predictions"])
#    recos = 1 / (1 + np.exp(-recos))
    ids = np.argpartition(recos, -count)[-count:]
    rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
    return list(
        islice(
            ((int(rec[0]), float(rec[1])) for rec in rank
             if rec[0] not in u_consumed), n_rec
        )
    )


def get_recommend_indices_and_values(user):
    if isinstance(user, str):
        user = int(user)
    user_indices = np.repeat(user, n_items)
    item_indices = np.arange(n_items)
    user_sparse_part = np.tile(user_sparse_unique[user], (n_items, 1))
    item_sparse_part = item_sparse_unique
    sparse_indices = np.concatenate(
        [user_sparse_part, item_sparse_part], axis=1)[:, sparse_col_reindex]
    dense_values = np.tile(user_dense_unique[user], (n_items, 1))
    return user_indices, item_indices, sparse_indices, dense_values


def get_user_last_interacted(user):
    user_info = np.asarray(
        json.loads(r.hget("user_last_interacted", user)), dtype=np.int32)
    length = len(user_info)
    if length < 10:
        u_last_interacted = np.zeros(10, dtype=np.int32)
        u_last_interacted[:length] = user_info
        u_last_interacted = np.tile(u_last_interacted, (n_items, 1))
    else:
        u_last_interacted = np.tile(user_info, (n_items, 1))
    u_interacted_len = np.repeat(length, n_items)
    return u_last_interacted, u_interacted_len


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': f"Bad Request: {request.url}. "
                   f"Please provide more data information...",
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp


# export FLASK_APP=tf_deploy.py
# export FLASK_ENV=development
# flask run
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # host="0.0.0.0"

