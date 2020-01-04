import requests
import json
import argparse
import sys, os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="request")
    parser.add_argument("--user", "-u", help="user")
    parser.add_argument("--data", "-d", type=str, help="sample data")
    parser.add_argument("--n_rec", "-n", type=int, default=7, help="num of recommendations")
    parser.add_argument("--host", default="localhost", help="host address")
    parser.add_argument("--port", default=5000, help="port")
    parser.add_argument("--algo", default=None, type=str, help="specific algorithm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()         # http://127.0.0.1:5000/predict
    if args.data is not None:
        url_predict = "http://{host}:{port}/{algo}/predict".format(host=args.host, port=args.port, algo=args.algo)
    #    url = "http://{host}/{algo}".format(host=args.host,  algo=args.algo)
        data_dict = json.loads(args.data)
        data = {"user": data_dict["user"],
                "item": data_dict["item"],
                "sex": data_dict["sex"],
                "age": data_dict["age"],
                "occupation": data_dict["occupation"],
                "title": data_dict["title"],
                "genre1": data_dict["genre1"],
                "genre2": data_dict["genre2"],
                "genre3": data_dict["genre3"]}

        try:
            response = requests.post(url_predict, data=json.dumps(data))
            print(response.json())
        except json.decoder.JSONDecodeError:
            print("Oh noooooooooooooo, you should choose algorithm wisely...")

    if args.user is not None:
        users = json.loads(args.user)
        for u in users:
            url_recommend = "http://{host}:{port}/{algo}/recommend".format(host=args.host, port=args.port,
                                                                           algo=args.algo)
            user = {"user": [u], "n_rec": [args.n_rec]}
            try:
                response = requests.post(url_recommend, data=json.dumps(user))
                print(response.json())
            except json.decoder.JSONDecodeError:
                print("Oh noooooooooooooo, you should choose algorithm wisely...")


#  python deploy_feat_request.py --data '{"user": [1], "item": [1193], "sex": ["F"], "age": [1],
#  "occupation": [10], "title": [2452.0], "genre1": ["Drama"], "genre2": ["missing"], "genre3": ["missing"]}' --algo FM

# python deploy_feat_request.py --data '{"user": [1, 1], "item": [1193, 1193], "sex": ["F", "F"], "age": [1, 1],
# "occupation": [10, 10], "title": [2452.0, 2452.0], "genre1": ["Drama", "Drama"], "genre2": ["missing", "missing"],
# "genre3": ["missing", "missing"]}' --algo FM --user [1, 2, 3] --n_rec 5




