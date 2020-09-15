import argparse
import json
import requests
from serving.flask import colorize


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean liked value expected...")


def parse_args():
    parser = argparse.ArgumentParser(description="request")
    parser.add_argument("--user", type=str, help="user index")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--n_rec", type=int, default=10,
                        help="num of recommendations")
    parser.add_argument("--port", default=5000, help="port")
    parser.add_argument("--algo", default="tf", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()         # http://127.0.0.1:5000/predict
    url = f"http://{args.host}:{args.port}/{args.algo}/recommend"
    data = {"user": args.user, "n_rec": args.n_rec}
    try:
        response = requests.post(url, json=json.dumps(data))
        response_str = f"request_json: {response.json()}"
        print(f"{colorize(response_str, 'green', bold=True)}")
    except TypeError:
        print("Could not serialize to json format...")
    except json.JSONDecodeError:
        print("Can't print response as json format...")

#  python tf_request.py --user 1 --n_rec 10

