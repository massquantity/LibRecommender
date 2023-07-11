from pathlib import Path

import pandas as pd
import tensorflow as tf

import libreco
from libreco.algorithms import Caser, RNN4Rec
from libreco.data import DatasetPure, split_by_ratio_chrono
from libreco.tfops import TF_VERSION

if __name__ == "__main__":
    print(f"tensorflow version: {TF_VERSION}")
    print(libreco)
    from libreco.algorithms._als import als_update
    from libreco.utils._similarities import forward_cosine, invert_cosine

    print("Cython functions: ", invert_cosine, forward_cosine, als_update)

    cur_path = Path(".").parent
    if Path.exists(cur_path / "sample_movielens_rating.dat"):
        data_path = cur_path / "sample_movielens_rating.dat"
    else:
        data_path = cur_path / "sample_data" / "sample_movielens_rating.dat"

    pd_data = pd.read_csv(data_path, sep="::", names=["user", "item", "label", "time"])
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)

    rnn = RNN4Rec(
        "ranking",
        data_info,
        rnn_type="lstm",
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=1,
        lr=0.001,
        lr_decay=False,
        hidden_units=(16, 16),
        reg=None,
        batch_size=256,
        num_neg=1,
        dropout_rate=None,
        recent_num=10,
        tf_sess_config=None,
    )
    rnn.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=[
            "loss",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "map",
            "ndcg",
        ],
        num_workers=2,
    )
    print("prediction: ", rnn.predict(user=1, item=2))
    print("recommendation: ", rnn.recommend_user(user=1, n_rec=7))

    tf.compat.v1.reset_default_graph()
    caser = Caser(
        "ranking",
        data_info=data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        batch_size=2048,
    )
    caser.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=[
            "loss",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "map",
            "ndcg",
        ],
        num_workers=2,
    )
