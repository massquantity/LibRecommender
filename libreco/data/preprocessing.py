import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer
)


def preprocess_data(data, dense_col=None, normalizer="min_max",
                    transformer=("log", "sqrt", "square")):

    if not dense_col:
        print("nothing to preprocessing...")
        return data

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
                d[dense_col] = scaler.fit_transform(
                    d[dense_col]).astype(np.float32)
            else:
                d[dense_col] = scaler.transform(
                    d[dense_col]).astype(np.float32)

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

