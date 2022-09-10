import time

import pandas as pd
import torch

from examples.utils import reset_state
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import (
    SVD,
    SVDpp,
    NCF,
    ALS,
    BPR,
    UserCF,
    ItemCF,
    RNN4Rec,
    Caser,
    WaveNet,
    Item2Vec,
    DeepWalk,
    NGCF,
    LightGCN,
)


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
    # do negative sampling, assume the data only contains positive feedback
    train_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2020
    )
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

    reset_state("SVD")
    svd = SVD(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=3,
        lr=0.001,
        reg=None,
        batch_size=256,
        num_neg=1,
    )
    svd.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", svd.predict(user=1, item=2333))
    print("recommendation: ", svd.recommend_user(user=1, n_rec=7))

    reset_state("SVD++")
    svdpp = SVDpp(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=3,
        lr=0.001,
        reg=None,
        batch_size=256,
    )
    svdpp.fit(
        train_data,
        verbose=2,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", svdpp.predict(user=1, item=2333))
    print("recommendation: ", svdpp.recommend_user(user=1, n_rec=7))

    reset_state("NCF")
    ncf = NCF(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=1,
        lr=0.001,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        hidden_units="128,64,32",
        tf_sess_config=None,
    )
    ncf.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", ncf.predict(user=1, item=2333))
    print("recommendation: ", ncf.recommend_user(user=1, n_rec=7))

    reset_state("ALS")
    als = ALS(
        "ranking",
        data_info,
        embed_size=16,
        n_epochs=2,
        reg=5.0,
        alpha=10,
        use_cg=False,
        n_threads=1,
        seed=42,
    )
    als.fit(
        train_data,
        verbose=2,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", als.predict(user=1, item=2333))
    print("recommendation: ", als.recommend_user(user=1, n_rec=7))

    reset_state("BPR")
    bpr = BPR(
        "ranking",
        data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=3,
        lr=3e-4,
        reg=None,
        batch_size=256,
        num_neg=1,
        use_tf=True,
    )
    bpr.fit(
        train_data,
        verbose=2,
        num_threads=4,
        eval_data=eval_data,
        metrics=metrics,
        optimizer="adam",
    )

    reset_state("RNN4Rec")
    rnn = RNN4Rec(
        "ranking",
        data_info,
        rnn_type="gru",
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=0.001,
        lr_decay=None,
        hidden_units="16,16",
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=None,
        recent_num=10,
        tf_sess_config=None,
    )
    rnn.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", rnn.predict(user=1, item=2333))
    print("recommendation: ", rnn.recommend_user(user=1, n_rec=7))

    reset_state("Caser")
    caser = Caser(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=None,
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.0,
        use_bn=False,
        nh_filters=16,
        nv_filters=4,
        recent_num=10,
        tf_sess_config=None,
    )
    caser.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", caser.predict(user=1, item=2333))
    print("recommendation: ", caser.recommend_user(user=1, n_rec=7))

    reset_state("WaveNet")
    wave = WaveNet(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=None,
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=0.0,
        use_bn=False,
        n_filters=16,
        n_blocks=2,
        n_layers_per_block=4,
        recent_num=10,
        tf_sess_config=None,
    )
    wave.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", wave.predict(user=1, item=2333))
    print("recommendation: ", wave.recommend_user(user=1, n_rec=7))

    reset_state("Item2Vec")
    item2vec = Item2Vec(
        "ranking",
        data_info,
        embed_size=16,
        norm_embed=False,
        window_size=3,
        n_epochs=2,
        n_threads=0,
    )
    item2vec.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", item2vec.predict(user=1, item=2333))
    print("recommendation: ", item2vec.recommend_user(user=1, n_rec=7))

    reset_state("DeepWalk")
    deepwalk = DeepWalk(
        "ranking",
        data_info,
        embed_size=16,
        norm_embed=False,
        n_walks=10,
        walk_length=10,
        window_size=5,
        n_epochs=2,
        n_threads=0,
    )
    deepwalk.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", deepwalk.predict(user=1, item=2333))
    print("recommendation: ", deepwalk.recommend_user(user=1, n_rec=7))

    reset_state("NGCF")
    ngcf = NGCF(
        "ranking",
        data_info,
        embed_size=16,
        n_epochs=2,
        lr=3e-4,
        lr_decay=None,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        node_dropout=0.0,
        message_dropout=0.0,
        hidden_units="64,64,64",
        device=torch.device("cpu"),
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

    reset_state("LightGCN")
    lightgcn = LightGCN(
        "ranking",
        data_info,
        embed_size=32,
        n_epochs=2,
        lr=1e-4,
        lr_decay=None,
        reg=0.0,
        batch_size=2048,
        num_neg=1,
        dropout=0.0,
        n_layers=3,
        device=torch.device("cpu"),
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

    reset_state("user_cf")
    user_cf = UserCF(task="ranking", data_info=data_info, k_sim=20, sim_type="cosine")
    user_cf.fit(
        train_data,
        verbose=2,
        mode="invert",
        num_threads=4,
        min_common=1,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", user_cf.predict(user=1, item=2333))
    print("recommendation: ", user_cf.recommend_user(user=1, n_rec=7))

    reset_state("item_cf")
    item_cf = ItemCF(task="ranking", data_info=data_info, k_sim=20, sim_type="pearson")
    item_cf.fit(
        train_data,
        verbose=2,
        mode="invert",
        num_threads=1,
        min_common=1,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", item_cf.predict(user=1, item=2333))
    print("recommendation: ", item_cf.recommend_user(user=1, n_rec=7))

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
