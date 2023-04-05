from pathlib import Path

import pandas as pd
import pytest

from libreco.data import DatasetPure, split_by_ratio_chrono


def test_build_negatives_deprecated():
    data_path = Path(__file__).parent / "sample_data" / "sample_movielens_rating.dat"
    pd_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    with pytest.deprecated_call():
        train_data.build_negative_samples(data_info)
    with pytest.deprecated_call():
        eval_data.build_negative_samples(data_info)
