import time
import numpy as np
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import SVD, SVDpp, NCF, ALS, UserCF, ItemCF, RNN4Rec

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
    data = pd.read_csv("sample_data/sample_movielens_rating.dat", sep="::",
                       names=["user", "item", "label", "time"])

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    print(data_info)

    reset_state("SVD")
    svd = SVD("rating", data_info, embed_size=16, n_epochs=3, lr=0.001,
              reg=None, batch_size=256, batch_sampling=False, num_neg=1)
    svd.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
            metrics=["rmse", "mae", "r2"])
    print("prediction: ", svd.predict(user=1, item=2333))
    print("recommendation: ", svd.recommend_user(user=1, n_rec=7))

    reset_state("SVD++")
    svdpp = SVDpp(task="rating", data_info=data_info, embed_size=16,
                  n_epochs=3, lr=0.001, reg=None, batch_size=256)
    svdpp.fit(train_data, verbose=2, eval_data=eval_data,
              metrics=["rmse", "mae", "r2"])
    print("prediction: ", svdpp.predict(user=1, item=2333))
    print("recommendation: ", svdpp.recommend_user(user=1, n_rec=7))

    reset_state("NCF")
    ncf = NCF("rating", data_info, embed_size=16, n_epochs=3, lr=0.001,
              lr_decay=False, reg=None, batch_size=256, num_neg=1, use_bn=True,
              dropout_rate=None, hidden_units="128,64,32", tf_sess_config=None)
    ncf.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
            metrics=["rmse", "mae", "r2"])
    print("prediction: ", ncf.predict(user=1, item=2333))
    print("recommendation: ", ncf.recommend_user(user=1, n_rec=7))

    reset_state("RNN4Rec")
    rnn = RNN4Rec("rating", data_info, rnn_type="lstm", loss_type="cross_entropy",
                  embed_size=16, n_epochs=2, lr=0.001, lr_decay=None,
                  hidden_units="16,16", reg=None, batch_size=256, num_neg=1,
                  dropout_rate=None, recent_num=10, tf_sess_config=None)
    rnn.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
            metrics=["rmse", "mae", "r2"])
    print("prediction: ", rnn.predict(user=1, item=2333))
    print("recommendation: ", rnn.recommend_user(user=1, n_rec=7))

    reset_state("ALS")
    als = ALS(task="rating", data_info=data_info, embed_size=16, n_epochs=2,
              reg=5.0, alpha=10, seed=42)
    als.fit(train_data, verbose=2, use_cg=False, n_threads=1,
            eval_data=eval_data, metrics=["rmse", "mae", "r2"])
    print("prediction: ", als.predict(user=1, item=2333))
    print("recommendation: ", als.recommend_user(user=1, n_rec=7))

    reset_state("user_cf")
    user_cf = UserCF(task="rating", data_info=data_info, k=20, sim_type="cosine")
    user_cf.fit(train_data, verbose=2, mode="invert", num_threads=4, min_common=1,
                eval_data=eval_data, metrics=["rmse", "mae", "r2"])
    print("prediction: ", user_cf.predict(user=1, item=2333))
    print("recommendation: ", user_cf.recommend_user(user=1, n_rec=7))

    reset_state("item_cf")
    item_cf = ItemCF(task="rating", data_info=data_info, k=20, sim_type="pearson")
    item_cf.fit(train_data, verbose=2, mode="invert", num_threads=1, min_common=1,
                eval_data=eval_data, metrics=["rmse", "mae", "r2"])
    print("prediction: ", item_cf.predict(user=1, item=2333))
    print("recommendation: ", item_cf.recommend_user(user=1, n_rec=7))

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
