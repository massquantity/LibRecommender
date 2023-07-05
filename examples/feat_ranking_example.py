import time

import pandas as pd
import tensorflow as tf

from libreco.algorithms import (
    DIN,
    FM,
    AutoInt,
    DeepFM,
    GraphSage,
    GraphSageDGL,
    PinSage,
    PinSageDGL,
    TwoTower,
    WideDeep,
    YouTubeRanking,
    YouTubeRetrieval,
)
from libreco.data import DatasetFeat, split_by_ratio_chrono


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


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

    reset_state("GraphSage")
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
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", graphsage.predict(user=1, item=2333))
    print("recommendation: ", graphsage.recommend_user(user=1, n_rec=7))
    print("batch recommendation: ", graphsage.recommend_user(user=[1, 2, 3], n_rec=7))

    reset_state("GraphSageDGL")
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
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", graphsage_dgl.predict(user=1, item=2333))
    print("recommendation: ", graphsage_dgl.recommend_user(user=1, n_rec=7))

    reset_state("PinSage")
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
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", pinsage.predict(user=1, item=2333))
    print("recommendation: ", pinsage.recommend_user(user=1, n_rec=7))
    print("batch recommendation: ", pinsage.recommend_user(user=[1, 2, 3], n_rec=7))

    reset_state("PinSageDGL")
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
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", pinsage_dgl.predict(user=1, item=2333))
    print("recommendation: ", pinsage_dgl.recommend_user(user=1, n_rec=7))

    reset_state("FM")
    fm = FM(
        "ranking",
        data_info,
        loss_type="cross_entropy",
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
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", fm.predict(user=1, item=2333))
    print("recommendation: ", fm.recommend_user(user=1, n_rec=7))

    reset_state("Wide_Deep")
    wd = WideDeep(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr={"wide": 0.01, "deep": 1e-4},
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
    )
    wd.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", wd.predict(user=1, item=2333))
    print("recommendation: ", wd.recommend_user(user=1, n_rec=7))

    reset_state("DeepFM")
    deepfm = DeepFM(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
    )
    deepfm.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", deepfm.predict(user=1, item=2333))
    print("recommendation: ", deepfm.recommend_user(user=1, n_rec=7))

    reset_state("TwoTower")
    two_tower = TwoTower(
        "ranking",
        data_info,
        loss_type="softmax",
        embed_size=16,
        norm_embed=True,
        n_epochs=2,
        lr=1e-3,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        use_correction=True,
        temperature=0.1,
        ssl_pattern=None,
        tf_sess_config=None,
    )
    two_tower.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", two_tower.predict(user=1, item=2333))
    print("recommendation: ", two_tower.recommend_user(user=1, n_rec=7))

    reset_state("AutoInt")
    autoint = AutoInt(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        att_embed_size=(4, 4),
        num_heads=2,
        use_residual=False,
        lr=1e-3,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        tf_sess_config=None,
    )
    autoint.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", autoint.predict(user=1, item=2333))
    print("recommendation: ", autoint.recommend_user(user=1, n_rec=7))

    reset_state("DIN")
    din = DIN(
        "ranking",
        data_info,
        loss_type="cross_entropy",
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
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
        use_tf_attention=True,
    )
    din.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", din.predict(user=1, item=2333))
    print("recommendation: ", din.recommend_user(user=1, n_rec=7))

    reset_state("YouTubeRanking")
    ytb_ranking = YouTubeRanking(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
    )
    ytb_ranking.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", ytb_ranking.predict(user=1, item=2333))
    print("recommendation: ", ytb_ranking.recommend_user(user=1, n_rec=7))

    # Since according to the paper, `YouTubeRetrieval` model can not use item features,
    # we provide an example here.
    reset_state("YouTuBeRetrieval")
    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    sparse_col = ["sex", "occupation"]  # no item feature
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = []  # empty item feature
    train_data, data_info = DatasetFeat.build_trainset(
        train_data,
        user_col,
        item_col,
        sparse_col,
        dense_col,
    )
    eval_data = DatasetFeat.build_testset(eval_data)

    ytb_retrieval = YouTubeRetrieval(
        "ranking",
        data_info,
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_sampled_per_batch=None,
        use_bn=False,
        dropout_rate=None,
        loss_type="sampled_softmax",
        hidden_units=(128, 64, 32),
        sampler="uniform",
        tf_sess_config=None,
    )
    ytb_retrieval.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
