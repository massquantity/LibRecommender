import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


def process_data(
    data, dense_col=None, normalizer="min_max", transformer=("log", "sqrt", "square")
):
    if not isinstance(dense_col, list):
        raise ValueError("dense_col must be a list...")

    if normalizer.lower() == "min_max":
        scaler = MinMaxScaler()
    elif normalizer.lower() == "standard":
        scaler = StandardScaler()
    elif normalizer.lower() == "robust":
        scaler = RobustScaler()
    elif normalizer.lower() == "power":
        scaler = PowerTransformer()
    else:
        raise ValueError("unknown normalize type...")

    dense_col_transformed = dense_col.copy()
    if isinstance(data, (list, tuple)):
        for i, d in enumerate(data):
            if i == 0:  # assume train_data is the first one
                d[dense_col] = scaler.fit_transform(d[dense_col]).astype(np.float32)
            else:
                d[dense_col] = scaler.transform(d[dense_col]).astype(np.float32)

            for col in dense_col:
                if d[col].min() < 0.0:
                    print("can't transform negative values...")
                    continue
                if transformer is not None:
                    if "log" in transformer:
                        name = col + "_log"
                        d[name] = np.log1p(d[col])
                        if i == 0:
                            dense_col_transformed.append(name)
                    if "sqrt" in transformer:
                        name = col + "_sqrt"
                        d[name] = np.sqrt(d[col])
                        if i == 0:
                            dense_col_transformed.append(name)
                    if "square" in transformer:
                        name = col + "_square"
                        d[name] = np.square(d[col])
                        if i == 0:
                            dense_col_transformed.append(name)

    else:
        data[dense_col] = scaler.fit_transform(data[dense_col])
        for col in dense_col:
            if data[col].min() < 0.0:
                print("can't transform negative values...")
                continue
            if transformer is not None:
                if "log" in transformer:
                    name = col + "_log"
                    data[name] = np.log1p(data[col])
                    dense_col_transformed.append(name)
                if "sqrt" in transformer:
                    name = col + "_sqrt"
                    data[name] = np.sqrt(data[col])
                    dense_col_transformed.append(name)
                if "square" in transformer:
                    name = col + "_square"
                    data[name] = np.square(data[col])
                    dense_col_transformed.append(name)

    return data, dense_col_transformed


def split_multi_value(
    data,
    multi_value_col,
    sep,
    max_len=None,
    pad_val="missing",
    user_col=None,
    item_col=None,
):
    """Transform multi-valued features to the divided sub-features.

    Parameters
    ----------
    data : pandas.DataFrame
        Original data.
    multi_value_col : list of str
        Multi-value columns names.
    sep : str
        Delimiter to use.
    max_len : list or tuple of int or None, default: None
        The maximum number of sub-features after transformation.
        If it is None, the maximum category length of all the samples will be used.
        If not None, it should be a list or tuple,
        because there are possibly many ``multi_value`` features.
    pad_val : Any or list of Any, default: "missing"
        The padding value used for missing features.
    user_col : list of str or None, default: None
        User column names.
    item_col : list of str or None, default: None
        Item column names.

    Returns
    -------
    data : pandas.DataFrame
        Transformed data.
    multi_sparse_col : list of str
        Transformed multi-sparse column names.
    user_sparse_col : list of str
        Transformed user columns.
    item_sparse_col : list of str
        Transformed item columns.

    Raises
    ------
    AssertionError
        If ``max_len`` is not list or tuple.
    AssertionError
        If ``max_len`` size != ``multi_value_col`` size.
    """
    if max_len is not None:
        assert isinstance(max_len, (list, tuple)), "`max_len` must be list or tuple"
        assert len(max_len) == len(multi_value_col), (
            "`max_len` must have same length as `multi_value_col`"
        )  # fmt: skip

    if not isinstance(pad_val, (list, tuple)):
        pad_val = [pad_val] * len(multi_value_col)
    assert len(multi_value_col) == len(pad_val), (
        "length of `multi_sparse_col` and `pad_val` doesn't match"
    )  # fmt: skip

    user_sparse_col, item_sparse_col, multi_sparse_col = [], [], []
    for j, col in enumerate(multi_value_col):
        sparse_col = []
        data[col] = (
            data[col]
            .str.strip(sep + " ")
            .str.replace("\\s+", "", regex=True)
            .str.lower()
        )
        data.loc[data[col] == "", col] = pad_val[j]
        split_col = data[col].str.split(sep)
        col_len = int(split_col.str.len().max()) if max_len is None else max_len[j]
        for i in range(col_len):
            new_col_name = col + f"_{i+1}"
            sparse_col.append(new_col_name)
            data[new_col_name] = split_col.str.get(i)
            data[new_col_name] = data[new_col_name].fillna(pad_val[j])

        multi_sparse_col.append(sparse_col)
        if user_col is not None and col in user_col:
            user_sparse_col.extend(sparse_col)
        elif item_col is not None and col in item_col:
            item_sparse_col.extend(sparse_col)

    data = data.fillna(pad_val[0]).drop(multi_value_col, axis=1)
    return data, multi_sparse_col, user_sparse_col, item_sparse_col
