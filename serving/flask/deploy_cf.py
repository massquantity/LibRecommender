from collections import defaultdict
import json
from flask import Flask, jsonify, request
import redis
from colorize import colorize


app = Flask(__name__)

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
k_sims = r.get("k_sims")
k_sims = json.loads(k_sims)
user_consumed = r.get("user_consumed")
user_consumed = json.loads(user_consumed)


# def prepare_redis()


@app.route("/<algo>", methods=['POST'])
def api_call(algo):
    test_json = request.get_json(force=True)
    test_json_str = f"test_json: {test_json}"
    print(f"{colorize(test_json_str, 'magenta', bold=True)}")

    try:
        test_data = json.loads(test_json)
        user = test_data["user"]
        n_rec = test_data["n_rec"]
        k_nbs = test_data["k_neighbors"]
    except json.JSONDecodeError:
        print("Could not parse json file...")
        raise
    except KeyError:
        return bad_request()

    reco_list = recommend(user, k_nbs, n_rec)
    response = jsonify({f'recommend list for user ({user})': reco_list})
    return response


def recommend(user, k, n_rec):
    u_consumed = set(user_consumed[user])
    result = defaultdict(lambda: 0.0)
    for i, i_label in user_consumed[user].items():
        for j, sim in k_sims[i][:k]:
            if j in u_consumed:
                continue
            result[j] += sim * i_label

    if len(result) == 0:
        return -1
    rank_items = [(k, v) for k, v in result.items()]
    return sorted(rank_items, key=lambda x: -x[1])[:n_rec]


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

