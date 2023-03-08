import time

import pandas as pd

from libreco.algorithms import (
    NGCF,
    GraphSage,
    GraphSageDGL,
    LightGCN,
    PinSage,
    PinSageDGL,
)
from libreco.data import DatasetFeat, DatasetPure, split_by_ratio_chrono

if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv(
        "sample_data/sample_movielens_rating.dat",
        sep="::",
        names=["user", "item", "label", "time"],
    )

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    print(data_info)

    # only do negative sampling on eval data
    eval_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2222
    )

    metrics = [
        "loss",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "map",
        "ndcg",
    ]

    print("\n", "=" * 30, "NGCF", "=" * 30)
    ngcf = NGCF(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=False,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        node_dropout=0.0,
        message_dropout=0.0,
        hidden_units=(64, 64, 64),
        device="cuda",
    )
    ngcf.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", ngcf.predict(user=1, item=2333))
    print("recommendation: ", ngcf.recommend_user(user=1, n_rec=7))
    print("batch recommendation: ", ngcf.recommend_user(user=[1, 2, 3], n_rec=7))

    print("\n", "=" * 30, "LightGCN", "=" * 30)
    lightgcn = LightGCN(
        "ranking",
        data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=False,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.0,
        n_layers=3,
        device="cuda",
    )
    lightgcn.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", lightgcn.predict(user=1, item=2333))
    print("recommendation: ", lightgcn.recommend_user(user=1, n_rec=7))
    print("batch recommendation: ", lightgcn.recommend_user(user=[1, 2, 3], n_rec=7))

    # use feat data in GraphSage and PinSage
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)

    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    # only do negative sampling on eval data
    eval_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2222
    )

    print("\n", "=" * 30, "GraphSage", "=" * 30)
    graphsage = GraphSage(
        "ranking",
        data_info,
        loss_type="max_margin",
        paradigm="i2i",
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.0,
        num_layers=1,
        num_neighbors=10,
        num_walks=10,
        sample_walk_len=5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
    )
    graphsage.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", graphsage.predict(user=1, item=2333))
    print("recommendation: ", graphsage.recommend_user(user=1, n_rec=7))
    print("batch recommendation: ", graphsage.recommend_user(user=[1, 2, 3], n_rec=7))

    print("\n", "=" * 30, "GraphSageDGL", "=" * 30)
    graphsage_dgl = GraphSageDGL(
        "ranking",
        data_info,
        loss_type="focal",
        paradigm="u2i",
        aggregator_type="gcn",
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.0,
        remove_edges=False,
        num_layers=2,
        num_neighbors=3,
        num_walks=10,
        sample_walk_len=5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
    )
    graphsage_dgl.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", graphsage_dgl.predict(user=1, item=2333))
    print("recommendation: ", graphsage_dgl.recommend_user(user=1, n_rec=7))

    print("\n", "=" * 30, "PinSage", "=" * 30)
    pinsage = PinSage(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        paradigm="u2i",
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.0,
        remove_edges=False,
        num_layers=1,
        num_neighbors=10,
        num_walks=10,
        neighbor_walk_len=2,
        sample_walk_len=5,
        termination_prob=0.5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
    )
    pinsage.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", pinsage.predict(user=1, item=2333))
    print("recommendation: ", pinsage.recommend_user(user=1, n_rec=7))
    print("batch recommendation: ", pinsage.recommend_user(user=[1, 2, 3], n_rec=7))

    print("\n", "=" * 30, "PinSageDGL", "=" * 30)
    pinsage_dgl = PinSageDGL(
        "ranking",
        data_info,
        loss_type="max_margin",
        paradigm="i2i",
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=3,
        dropout_rate=0.0,
        remove_edges=False,
        num_layers=2,
        num_neighbors=3,
        num_walks=10,
        neighbor_walk_len=2,
        sample_walk_len=5,
        termination_prob=0.5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
    )
    pinsage_dgl.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", pinsage_dgl.predict(user=1, item=2333))
    print("recommendation: ", pinsage_dgl.recommend_user(user=1, n_rec=7))

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
