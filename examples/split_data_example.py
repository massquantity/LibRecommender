import time
import numpy as np
import pandas as pd
from libreco.data import (
    split_by_num,
    split_by_ratio,
    split_by_num_chrono,
    split_by_ratio_chrono,
    random_split
)


if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv("sample_data/sample_movielens_merged.csv",
                       sep=",", header=0)

    train_data, eval_data, test_data = random_split(
        data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, eval_data = split_by_ratio(data, test_size=0.2)
    print(train_data.shape, eval_data.shape)

    train_data, eval_data = split_by_num(data, test_size=1)
    print(train_data.shape, eval_data.shape)

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    print(train_data.shape, eval_data.shape)

    train_data, eval_data = split_by_num_chrono(data, test_size=1)
    print(train_data.shape, eval_data.shape)
