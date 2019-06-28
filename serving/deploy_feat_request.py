import requests
import json
import argparse
import sys, os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="request")
    parser.add_argument("--data", "-d", type=str, help="sample data")
    parser.add_argument("--host", default="localhost", help="host address")
    parser.add_argument("--port", default=5000, help="port")
    parser.add_argument("--algo", default=None, type=str, help="specific algorithm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()         # http://127.0.0.1:5000/predict
    url = "http://{host}:{port}/{algo}".format(host=args.host, port=args.port, algo=args.algo)
    # if os.path.isfile(args.data)  elif isinstance(args.data, str)
    data_dict = json.loads(args.data)
    data = {"user": [data_dict["user"]],
            "item": [data_dict["item"]],
            "sex": [data_dict["sex"]],
            "occupation": [data_dict["occupation"]],
            "title": [data_dict["title"]],
            "genre1": [data_dict["genre1"]],
            "genre2": [data_dict["genre2"]],
            "genre3": [data_dict["genre3"]],
        }

    try:
        response = requests.post(url, data=json.dumps(data))   #   json={'u': [1], 'i': [3]}
        print(response.json())
    except json.decoder.JSONDecodeError:
        print("Oh noooooooooooooo, you should choose algorithm wisely...")

#  python deploy_pure_request.py --user 1 --item 3 -k 10 -n 10

#  python deploy_feat_request.py --data '{"user": 1, "item": 1193, "sex": "F", "age": 1,
#  "occupation": 10, "title": 2452.0, "genre1": "Drama", "genre2": "missing", "genre3": "missing"}' --algo FM