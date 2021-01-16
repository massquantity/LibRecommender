import numpy as np
import pandas as pd
from libreco.data import DatasetFeat, DataInfo
from libreco.data import split_by_ratio_chrono
from libreco.algorithms import DeepFM

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["KMP_WARNINGS"] = "FALSE"


if __name__ == "__main__":
    col_names = ["user", "item", "label", "time", "sex",
                 "age", "occupation", "genre1", "genre2", "genre3"]
    data = pd.read_csv("sample_data/sample_movielens_merged.csv",
                       sep=",", header=0)
    train, test = split_by_ratio_chrono(data, test_size=0.2)

    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]
    train_data, data_info = DatasetFeat.build_trainset(train, user_col, item_col,
                                                       sparse_col, dense_col,
                                                       shuffle=False)
    test_data = DatasetFeat.build_testset(test, shuffle=False)
    print(data_info)
    train_data.build_negative_samples(data_info, num_neg=1,
                                      item_gen_mode="random", seed=2020)
    test_data.build_negative_samples(data_info, num_neg=1,
                                     item_gen_mode="random", seed=2222)

    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                    lr=1e-4, lr_decay=False, reg=None, batch_size=2048,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=test_data,
               metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                        "precision", "recall", "map", "ndcg"],
               eval_batch_size=8192, k=10, sample_user_num=2048)

    print("prediction: ", deepfm.predict(1, 110))
    print("recommendation: ", deepfm.recommend_user(1, 7))

    # save data_info, specify model save folder
    data_info.save("model_path")
    # set manual=True will use numpy to save model
    # set manual=False will use tf.train.Saver to save model
    deepfm.save("model_path", "deepfm_model", manual=True)

    print("\n", "="*50, " after load model ", "="*50)
    # load data_info
    data_info = DataInfo.load("model_path")
    print(data_info)
    # load model, should specify the model name, e.g., DeepFM
    deepfm = DeepFM.load("model_path", "deepfm_model", data_info, manual=True)
    print("eval result: ", deepfm.evaluate(test_data, metrics=["precision"]))

    print("prediction: ", deepfm.predict(1, 110))
    print("recommendation: ", deepfm.recommend_user(1, 7))
