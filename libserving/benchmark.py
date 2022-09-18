import argparse
import asyncio
import time

import aiohttp
import requests


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000)
    parser.add_argument("--user", type=str, help="user id")
    parser.add_argument("--n_rec", type=int, help="num of recommendations")
    parser.add_argument("--n_times", type=int, help="num of requests")
    parser.add_argument("--algo", type=str, help="type of algorithm")
    return parser.parse_args()


async def get_reco(session: aiohttp.ClientSession, url: str, data: dict):
    async with session.post(url, json=data) as resp:
        reco = await resp.json()
    return reco


async def main_async(args):
    # noinspection HttpUrlsUsage
    url = f"http://{args.host}:{args.port}/{args.algo}/recommend"
    data = {"user": args.user, "n_rec": args.n_rec}
    async with aiohttp.ClientSession() as session:
        tasks = [get_reco(session, url, data) for _ in range(args.n_times)]
        await asyncio.gather(*tasks)


def main_sync(args):
    # noinspection HttpUrlsUsage
    url = f"http://{args.host}:{args.port}/{args.algo}/recommend"
    data = {"user": args.user, "n_rec": args.n_rec}
    for _ in range(args.n_times):
        resp = requests.post(url, json=data)
        _ = resp.json()


if __name__ == "__main__":
    args = parse_args()

    start = time.perf_counter()
    asyncio.run(main_async(args))
    duration = time.perf_counter() - start
    print(
        f"total time {duration}s for async requests, "
        f"{duration / args.n_times * 1000} ms/request"
    )

    start = time.perf_counter()
    main_sync(args)
    duration = time.perf_counter() - start
    print(
        f"total time {duration}s for sync requests, "
        f"{duration / args.n_times * 1000} ms/request"
    )