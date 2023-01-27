from io import StringIO

import pandas as pd

from libreco.data import (
    random_split,
    split_by_num,
    split_by_num_chrono,
    split_by_ratio,
    split_by_ratio_chrono,
)

raw_data = StringIO(
    """
user,item,label,time
4617,296,2,964138229
4617,296,2,964138221
4617,296,2,964138222
1298,208,4,974849526
4585,1769,4,964322774
3706,1136,5,966376465
2137,1215,3,974640099
4617,208,4,974170662
1298,1769,3,977854274
4585,208,4,974607346
263,1136,2,976651827
5184,1215,5,961735308
"""
)
# noinspection PyTypeChecker
pd_data = pd.read_csv(raw_data, header=0)


def test_random_split():
    split_data = random_split(pd_data, multi_ratios=[0.8, 0.1, 0.1])
    assert len(split_data) == 3

    train_data, eval_data = random_split(
        pd_data, test_size=0.5, shuffle=False, filter_unknown=True
    )
    assert len(eval_data) == 3

    train_data, eval_data = random_split(
        pd_data,
        test_size=0.5,
        shuffle=False,
        filter_unknown=False,
        pad_unknown=True,
        pad_val=[-1, -1],
    )
    assert len(eval_data) == 6
    assert eval_data["user"].min() == -1

    train_data, eval_data = random_split(
        pd_data,
        test_size=0.5,
        shuffle=False,
        filter_unknown=False,
        pad_unknown=True,
        pad_val=0,
    )
    assert len(eval_data) == 6
    print(eval_data["user"])
    assert eval_data["user"].min() == 0


def test_split_by_ratio():
    train_data, eval_data = split_by_ratio(pd_data, test_size=0.5, filter_unknown=True)
    assert len(train_data) == 10
    assert len(eval_data) == 2
    train_data, eval_data = split_by_ratio(
        pd_data,
        test_size=0.5,
        shuffle=True,
        filter_unknown=False,
        pad_unknown=True,
        pad_val=0,
    )
    assert len(train_data) == 10
    assert len(eval_data) == 2


def test_split_by_num():
    train_data, eval_data = split_by_num(pd_data, test_size=1, filter_unknown=True)
    assert len(train_data) == 11
    assert len(eval_data) == 1
    train_data, eval_data = split_by_num(
        pd_data,
        test_size=1,
        shuffle=True,
        filter_unknown=False,
        pad_unknown=True,
        pad_val=0,
    )
    assert len(train_data) == 11
    assert len(eval_data) == 1


def test_split_by_ratio_chrono():
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.5)
    assert len(train_data) == 10
    assert len(eval_data) == 2


def test_split_by_num_chrono():
    train_data, eval_data = split_by_num_chrono(pd_data, test_size=1)
    assert len(train_data) == 11
    assert len(eval_data) == 1
