import os

import pandas as pd
import pytest

from libreco.data import split_multi_value


def test_multi_sparse_processing():
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sample_data",
        "sample_movielens_genre.csv",
    )
    data = pd.read_csv(data_path, sep=",", header=0)

    with pytest.raises(AssertionError):
        # max_len must be list or tuple
        split_multi_value(data, multi_value_col=["genre"], sep="|", max_len=3)

    sep = ","  # wrong separator
    data, *_ = split_multi_value(
        data,
        multi_value_col=["genre"],
        sep=sep,
        max_len=[3],
        pad_val="missing",
        user_col=["sex", "age", "occupation"],
        item_col=["genre"],
    )
    assert all(data["genre_2"].str.contains("missing"))
    assert all(data["genre_3"].str.contains("missing"))

    sep = "|"
    data = pd.read_csv(data_path, sep=",", header=0)
    data, multi_sparse_col, multi_user_col, multi_item_col = split_multi_value(
        data,
        multi_value_col=["genre"],
        sep=sep,
        max_len=[3],
        pad_val="missing",
        user_col=["sex", "age", "occupation"],
        item_col=["genre"],
    )
    assert multi_sparse_col == [["genre_1", "genre_2", "genre_3"]]
    assert multi_user_col == []
    assert multi_item_col == ["genre_1", "genre_2", "genre_3"]
    all_columns = data.columns.tolist()
    assert "genre" not in all_columns
    assert all_columns == [
        "user",
        "item",
        "label",
        "time",
        "sex",
        "age",
        "occupation",
        "genre_1",
        "genre_2",
        "genre_3",
    ]
