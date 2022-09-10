import time

import pandas as pd

from examples.utils import reset_state
from libreco.data import split_by_ratio_chrono, DatasetFeat
from libreco.algorithms import FM, WideDeep, DeepFM, AutoInt, DIN


if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)

    # specify complete columns information
    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    metrics = ["rmse", "mae", "r2"]

    reset_state("FM")
    fm = FM(
        "rating",
        data_info,
        embed_size=16,
        n_epochs=3,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        tf_sess_config=None,
    )
    fm.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", fm.predict(user=1, item=2333))
    print("recommendation: ", fm.recommend_user(user=1, n_rec=7))

    reset_state("Wide_Deep")
    wd = WideDeep(
        "rating",
        data_info,
        embed_size=16,
        n_epochs=2,
        lr={"wide": 0.01, "deep": 1e-4},
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units="128,64,32",
        tf_sess_config=None,
    )
    wd.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", wd.predict(user=1, item=2333))
    print("recommendation: ", wd.recommend_user(user=1, n_rec=7))

    reset_state("DeepFM")
    deepfm = DeepFM(
        "rating",
        data_info,
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units="128,64,32",
        tf_sess_config=None,
    )
    deepfm.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", deepfm.predict(user=1, item=2333))
    print("recommendation: ", deepfm.recommend_user(user=1, n_rec=7))

    reset_state("AutoInt")
    autoint = AutoInt(
        "rating",
        data_info,
        embed_size=16,
        n_epochs=2,
        att_embed_size=(8, 8, 8),
        num_heads=4,
        use_residual=False,
        lr=1e-3,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units="128,64,32",
        tf_sess_config=None,
    )
    autoint.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", autoint.predict(user=1, item=2333))
    print("recommendation: ", autoint.recommend_user(user=1, n_rec=7))

    reset_state("DIN")
    din = DIN(
        "rating",
        data_info,
        embed_size=16,
        n_epochs=2,
        recent_num=10,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units="128,64,32",
        tf_sess_config=None,
        use_tf_attention=True,
    )
    din.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", din.predict(user=1, item=2333))
    print("recommendation: ", din.recommend_user(user=1, n_rec=7))

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
