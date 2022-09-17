import pandas as pd
import tensorflow as tf

from libreco.data import DatasetFeat, DataInfo
from libreco.data import split_by_ratio_chrono
from libreco.algorithms import DeepFM
from libreco.evaluation import evaluate


if __name__ == "__main__":
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)
    train, test = split_by_ratio_chrono(data, test_size=0.2)

    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]
    train_data, data_info = DatasetFeat.build_trainset(
        train, user_col, item_col, sparse_col, dense_col, shuffle=False
    )
    test_data = DatasetFeat.build_testset(test, shuffle=False)
    print(data_info)
    train_data.build_negative_samples(
        data_info, num_neg=1, item_gen_mode="random", seed=2020
    )
    test_data.build_negative_samples(
        data_info, num_neg=1, item_gen_mode="random", seed=2222
    )

    deepfm = DeepFM(
        "ranking",
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
        eval_data=test_data,
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
        eval_batch_size=8192,
        k=10,
        sample_user_num=2048,
    )

    print("prediction: ", deepfm.predict(user=2211, item=110))
    print("recommendation: ", deepfm.recommend_user(user=2211, n_rec=7))

    # save data_info, specify model save folder
    data_info.save(path="model_path", model_name="deepfm_model")
    # set manual=True will use numpy to save model
    # set manual=False will use tf.train.Saver to save model
    # set inference=True will only save the necessary variables for prediction and recommendation
    deepfm.save(
        path="model_path", model_name="deepfm_model", manual=True, inference_only=True
    )

    # =========================== load model ==============================
    print("\n", "=" * 50, " after load model ", "=" * 50)
    # important to reset graph if model is loaded in the same shell.
    tf.compat.v1.reset_default_graph()
    # load data_info
    data_info = DataInfo.load("model_path", model_name="deepfm_model")
    print(data_info)
    # load model, should specify the model name, e.g., DeepFM
    model = DeepFM.load(
        path="model_path", model_name="deepfm_model", data_info=data_info, manual=True
    )

    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)
    train, test = split_by_ratio_chrono(data, test_size=0.2)
    eval_result = evaluate(
        model=model,
        data=test,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "precision"],
        sample_user_num=2048,
        neg_sample=True,
        update_features=False,
        seed=2222,
    )
    print("eval result: ", eval_result)

    print("prediction: ", model.predict(user=2211, item=110))
    print("recommendation: ", model.recommend_user(user=2211, n_rec=7))
