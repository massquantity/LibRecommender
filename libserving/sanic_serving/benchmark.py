import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import requests
import ujson

REQUEST_LIMIT = 64


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000)
    parser.add_argument("--user", type=str, help="user id")
    parser.add_argument("--n_rec", type=int, help="num of recommendations")
    parser.add_argument("--n_times", type=int, help="num of requests")
    parser.add_argument("--n_threads", type=int, default=1, help="num of threads")
    parser.add_argument("--algo", type=str, help="type of algorithm")
    return parser.parse_args()


async def get_reco_async(
    session: aiohttp.ClientSession, url: str, data: dict, semaphore: asyncio.Semaphore
):
    async with semaphore, session.post(url, json=data) as resp:
        # if semaphore.locked():
        #     await asyncio.sleep(1.0)
        resp.raise_for_status()
        reco = await resp.json(loads=ujson.loads)
    return reco


async def main_async(args):
    url = f"http://{args.host}:{args.port}/{args.algo}/recommend"
    data = {"user": args.user, "n_rec": args.n_rec}
    semaphore = asyncio.Semaphore(REQUEST_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_reco_async(session, url, data, semaphore) for _ in range(args.n_times)
        ]
        # await asyncio.gather(*tasks, return_exceptions=True)
        for future in asyncio.as_completed(tasks):
            _ = await future


def get_reco_sync(url: str, data: dict):
    resp = requests.post(url, json=data, timeout=1)
    resp.raise_for_status()
    return resp.json()


def main_sync(args):
    url = f"http://{args.host}:{args.port}/{args.algo}/recommend"
    data = {"user": args.user, "n_rec": args.n_rec}
    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = [
            executor.submit(get_reco_sync, url, data) for _ in range(args.n_times)
        ]
        for future in as_completed(futures):
            _ = future.result()


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
