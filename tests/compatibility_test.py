import os

import pandas as pd

import libreco
from libreco.algorithms import UserCF, ALS, BPR, RNN4Rec
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.utils.similarities import cosine_sim


if __name__ == "__main__":
    print(libreco)
    print(UserCF, ALS, BPR)
    print(cosine_sim)

    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        # "sample_data",
        "sample_movielens_rating.dat"
    )
    pd_data = pd.read_csv(data_path, sep="::", names=["user", "item", "label", "time"])
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    train_data.build_negative_samples(data_info, item_gen_mode="random",
                                      num_neg=1, seed=2022)
    eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                     num_neg=1, seed=2222)

    rnn = RNN4Rec("ranking", data_info, rnn_type="lstm", loss_type="cross_entropy",
                  embed_size=16, n_epochs=1, lr=0.001, lr_decay=None,
                  hidden_units="16,16", reg=None, batch_size=256, num_neg=1,
                  dropout_rate=None, recent_num=10, tf_sess_config=None)
    rnn.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
            metrics=["loss", "balanced_accuracy",
                     "roc_auc", "pr_auc", "precision",
                     "recall", "map", "ndcg"])
    print("prediction: ", rnn.predict(user=1, item=2333))
    print("recommendation: ", rnn.recommend_user(user=1, n_rec=7))
