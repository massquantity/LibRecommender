import argparse
import json

import requests


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000)
    parser.add_argument("--user", type=str, help="user id")
    parser.add_argument("--n_rec", type=int, help="num of recommendations")
    parser.add_argument("--algo", type=str, help="type of serving algorithm")
    return parser.parse_args()


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}/{args.algo}/recommend"
    data = {"user": args.user, "n_rec": args.n_rec}
    response = requests.post(url, json=data, timeout=1)
    if response.status_code != 200:
        print(f"Failed to get recommendation: {url}")
        print(response.text)
        response.raise_for_status()
    try:
        print(response.json())
    except json.JSONDecodeError:
        print("Failed to decode response json")


if __name__ == "__main__":
    main()
