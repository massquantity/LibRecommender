import requests
import json
import argparse
import sys, os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="request")
    parser.add_argument("--user", "-u", type=int, help="user index")
    parser.add_argument("--item", "-i", type=int, help="item index")
    parser.add_argument("--host", default="localhost", help="host address")
    parser.add_argument("--k_neighbor", "-k", type=int, help="k nearest neighbors")
    parser.add_argument("--n_rec", "-n", type=int, help="num of recommendations")
    parser.add_argument("--port", default=5000, help="port")
    parser.add_argument("--algo", default=None, type=str, help="specific algorithm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()         # http://127.0.0.1:5000/predict
    url = "http://{host}:{port}/{algo}".format(host=args.host, port=args.port, algo=args.algo)
    data = {"u": [args.user], "i": [args.item], "k": [args.k_neighbor], "n_rec": [args.n_rec]}
    try:
        response = requests.post(url, data=json.dumps(data))   #   json={'u': [1], 'i': [3]}
        print(response.json())
    except json.decoder.JSONDecodeError:
        print("Oh noooooooooooooo, you should choose algorithm wisely...")

#  python deploy_pure_request.py --user 1 --item 3 -k 10 -n 10

