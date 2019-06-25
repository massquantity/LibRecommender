import requests
import json
import argparse
import sys, os
from pathlib import Path
sys.path.append(Path(os.getcwd()).parent.parent)
print(Path(os.getcwd()).parent.parent)



def parse_args():
    parser = argparse.ArgumentParser(description="request")
    parser.add_argument("--user", "-u", type=int, default=1, help="user index")
    parser.add_argument("--item", "-i", type=int, default=1, help="item index")
    parser.add_argument("--host", default="localhost", help="host address")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    url = "http://" + args.host + ":5000/predict"
    data = {"u": [args.user], "i": [args.item]}
    response = requests.post(url, data=json.dumps(data))   #   json={'u': [1], 'i': [3]}
    print(response.json())

#  python deploy_request.py --user 1 --item 3

