import pandas as pd
import tensorflow as tf

from libreco.data import DatasetFeat, DataInfo
from libreco.data import split_by_ratio_chrono
from libreco.algorithms import DeepFM
from libreco.evaluation import evaluate
from libreco.prediction import predict_data_with_feats


if __name__ == "__main__":
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)
    train, test = split_by_ratio_chrono(data, test_size=0.2)

    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]
    train_data, data_info = DatasetFeat.build_trainset(
        train,
        user_col,
        item_col,
        sparse_col,
        dense_col,
        shuffle=False,
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
    )

    data_info.save(path="model_path", model_name="deepfm_model")
    deepfm.save(
        path="model_path", model_name="deepfm_model", manual=True, inference_only=True
    )

    # =========================== load model ==============================
    print("\n", "=" * 50, " after load model ", "=" * 50)
    tf.compat.v1.reset_default_graph()
    data_info = DataInfo.load("model_path", model_name="deepfm_model")
    model = DeepFM.load(
        path="model_path", model_name="deepfm_model", data_info=data_info, manual=True
    )

    # predict with feats needed
    print(
        "prediction: ",
        model.predict(user=2211, item=110, feats={"sex": "mmm", "genre1": "crime"}),
    )
    print(
        "recommendation: ",
        model.recommend_user(
            user=2211,
            n_rec=7,
            inner_id=False,
            cold_start="average",
            user_feats=pd.Series({"sex": "F", "occupation": 2, "age": 23}),
            item_data=data.iloc[4:10],
        ),
    )
    print()

    # change some user to cold-start user
    data.iloc[2, 0] = -999
    print("changed data: \n", data.head())
    print(
        "prediction with data: ",
        predict_data_with_feats(
            deepfm, data.iloc[:5, :], batch_size=2, cold_start="average"
        ),
    )
    print()

    # assign features to DataInfo object
    data_info.assign_user_features(user_data=data)
    data_info.assign_item_features(item_data=data)

    # set neg_sample=True if data is implicit and only contains positive label
    eval_result = evaluate(
        model=model,
        data=test,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        sample_user_num=2048,
        neg_sample=True,
        update_features=False,
        seed=2222,
    )
    print("Eval Result: ")
    print(eval_result)
