import time
import numpy as np
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetFeat
from libreco.algorithms import (
    FM, WideDeep, DeepFM, AutoInt, DIN, YouTubeMatch, YouTubeRanking
)

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv("sample_data/sample_movielens_merged.csv",
                       sep=",", header=0)
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
    # do negative sampling, assume the data only contains positive feedback
    train_data.build_negative_samples(data_info, item_gen_mode="random",
                                      num_neg=1, seed=2020)
    eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                     num_neg=1, seed=2222)

    reset_state("FM")
    fm = FM("ranking", data_info, embed_size=16, n_epochs=3,
            lr=1e-4, lr_decay=False, reg=None, batch_size=256,
            num_neg=1, use_bn=True, dropout_rate=None, tf_sess_config=None)
    fm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
           metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                    "precision", "recall", "map", "ndcg"])
    print("prediction: ", fm.predict(user=1, item=2333))
    print("recommendation: ", fm.recommend_user(user=1, n_rec=7))

    reset_state("Wide_Deep")
    wd = WideDeep("ranking", data_info, embed_size=16, n_epochs=2,
                  lr={"wide": 0.01, "deep": 1e-4}, lr_decay=False, reg=None,
                  batch_size=256, num_neg=1, use_bn=False, dropout_rate=None,
                  hidden_units="128,64,32", tf_sess_config=None)
    wd.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
           metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                    "precision", "recall", "map", "ndcg"])
    print("prediction: ", wd.predict(user=1, item=2333))
    print("recommendation: ", wd.recommend_user(user=1, n_rec=7))

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                    lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                        "precision", "recall", "map", "ndcg"])
    print("prediction: ", deepfm.predict(user=1, item=2333))
    print("recommendation: ", deepfm.recommend_user(user=1, n_rec=7))

    reset_state("AutoInt")
    autoint = AutoInt("ranking", data_info, embed_size=16, n_epochs=2,
                      att_embed_size=(8, 8, 8), num_heads=4, use_residual=False,
                      lr=1e-3, lr_decay=False, reg=None, batch_size=2048,
                      num_neg=1, use_bn=False, dropout_rate=None,
                      hidden_units="128,64,32", tf_sess_config=None)
    autoint.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
                metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                         "precision", "recall", "map", "ndcg"])
    print("prediction: ", autoint.predict(user=1, item=2333))
    print("recommendation: ", autoint.recommend_user(user=1, n_rec=7))

    reset_state("DIN")
    din = DIN("ranking", data_info, embed_size=16, n_epochs=2,
              recent_num=10, lr=1e-4, lr_decay=False, reg=None,
              batch_size=2048, num_neg=1, use_bn=False, dropout_rate=None,
              hidden_units="128,64,32", tf_sess_config=None, use_tf_attention=True)
    din.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
            metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                     "precision", "recall", "map", "ndcg"])
    print("prediction: ", din.predict(user=1, item=2333))
    print("recommendation: ", din.recommend_user(user=1, n_rec=7))

    reset_state("YouTubeRanking")
    ytb_ranking = YouTubeRanking("ranking", data_info, embed_size=16, n_epochs=2,
                                 lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                                 num_neg=1, use_bn=False, dropout_rate=None,
                                 hidden_units="128,64,32", tf_sess_config=None)
    ytb_ranking.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
                    recent_num=None, sample_rate=None,
                    metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                             "precision", "recall", "map", "ndcg"])
    print("prediction: ", ytb_ranking.predict(user=1, item=2333))
    print("recommendation: ", ytb_ranking.recommend_user(user=1, n_rec=7))

    # Since according to the paper, the YouTubeMatch model can not use item features,
    # we provide an example here.
    reset_state("YouTubeMatch")
    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    sparse_col = ["sex", "occupation"]  # no item feature
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = []  # empty item feature
    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col,
    )
    eval_data = DatasetFeat.build_testset(eval_data)

    train_data.build_negative_samples(data_info, item_gen_mode="random",
                                      num_neg=1, seed=2020)
    eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                     num_neg=1, seed=2222)

    ytb_match = YouTubeMatch("ranking", data_info, embed_size=16, n_epochs=3,
                             lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                             num_neg=1, use_bn=False, dropout_rate=None,
                             loss_type="sampled_softmax", hidden_units="128,64,32",
                             tf_sess_config=None, sampler="uniform")
    ytb_match.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
                  metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                           "precision", "recall", "map", "ndcg"])

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
