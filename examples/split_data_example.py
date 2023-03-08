import time

import pandas as pd

from libreco.data import (
    random_split,
    split_by_num,
    split_by_num_chrono,
    split_by_ratio,
    split_by_ratio_chrono,
)

if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)

    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data2, eval_data2 = split_by_ratio(data, test_size=0.2)
    print(train_data2.shape, eval_data2.shape)

    train_data3, eval_data3 = split_by_num(data, test_size=1)
    print(train_data3.shape, eval_data3.shape)

    train_data4, eval_data4 = split_by_ratio_chrono(data, test_size=0.2)
    print(train_data4.shape, eval_data4.shape)

    train_data5, eval_data5 = split_by_num_chrono(data, test_size=1)
    print(train_data5.shape, eval_data5.shape)
