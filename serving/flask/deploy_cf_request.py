import argparse
import json
import time
import requests
from colorize import colorize


def parse_args():
    parser = argparse.ArgumentParser(description="request")
    parser.add_argument("--user", type=str, help="user index")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--k_neighbors", default=10, type=int)
    parser.add_argument("--n_rec", type=int, help="num of recommendations")
    parser.add_argument("--port", default=5000, help="port")
    parser.add_argument("--algo", default="item_cf", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()         # http://127.0.0.1:5000/predict
    url = f"http://{args.host}:{args.port}/{args.algo}"
    data = {"user": args.user, "n_rec": args.n_rec,
            "k_neighbors": args.k_neighbors}
    try:
        response = requests.post(url, json=json.dumps(data))
        response_str = f"request_json: {response.json()}"
        print(f"{colorize(response_str, 'green', bold=True)}")
    except TypeError:
        print("Could not serialize to json format...")

#  python deploy_pure_request.py --user 1 -k_neighbors 10 -n_rec 10

