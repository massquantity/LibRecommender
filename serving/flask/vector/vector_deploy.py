import ast
from itertools import islice
import json
from flask import Flask, jsonify, request
import faiss
import numpy as np
import redis
from serving.flask import colorize


app = Flask(__name__)


def get_item_vector_from_redis(name):
    item_vector_from_redis = r.hgetall(name)
    length = len(item_vector_from_redis)
    embed_size = len(ast.literal_eval(item_vector_from_redis['1']))
    item_vectors = np.zeros((length, embed_size), dtype=np.float32)
    for i in range(length):
        item_vectors[i] = json.loads(item_vector_from_redis[str(i)])
    return item_vectors


r = redis.Redis(host="localhost", port=6379, decode_responses=True)
user_consumed = r.get("user_consumed")
user_consumed = json.loads(user_consumed)
item_vector = get_item_vector_from_redis("item_vector")

# quantizer = faiss.IndexFlatL2(item_vector.shape[1])
# index = faiss.IndexIVFFlat(quantizer, item_vector.shape[1], 100)
# index.train(item_vector)
# index.add(item_vector)


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
        use_faiss = test_data["use_faiss"]
    except json.JSONDecodeError:
        print("Could not parse json file...")
        raise
    except KeyError:
        return bad_request()

    reco_list = recommend(user, n_rec, use_faiss)
    response = jsonify({f'recommend list for user ({user})': reco_list})
    return response


def recommend(user, n_rec, use_faiss):
    u_consumed = set(user_consumed[user])
    count = n_rec + len(u_consumed)
    user_vector = np.asarray(
        json.loads(
            r.hget("user_vector", user)
        ), dtype=np.float32
    )

    if use_faiss:
        index = faiss.read_index("vector_model/faiss_index.bin")
        _, recos = index.search(user_vector.reshape(1, -1), n_rec)
        return recos.flatten().tolist()
    else:
        recos = item_vector @ user_vector
    #    recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        return list(
            islice(
                ((int(rec[0]), float(rec[1])) for rec in rank
                 if rec[0] not in u_consumed), n_rec
            )
        )


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)  #  host="0.0.0.0"

