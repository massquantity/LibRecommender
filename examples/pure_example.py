import pandas as pd

from libreco.algorithms import LightGCN  # pure data, algorithm LightGCN
from libreco.data import DatasetPure, random_split
from libreco.evaluation import evaluate

if __name__ == "__main__":
    data = pd.read_csv(
        "sample_data/sample_movielens_rating.dat",
        sep="::",
        names=["user", "item", "label", "time"],
    )

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)
    print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

    lightgcn = LightGCN(
        task="ranking",
        data_info=data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=3,
        lr=1e-3,
        batch_size=2048,
        num_neg=1,
        device="cuda",
    )
    # monitor metrics on eval_data during training
    lightgcn.fit(
        train_data,
        neg_sampling=True,  # sample negative items for train and eval data
        verbose=2,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

    # do final evaluation on test data
    print(
        "evaluate_result: ",
        evaluate(
            model=lightgcn,
            data=test_data,
            neg_sampling=True,  # sample negative items for test data
            metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
        ),
    )
    # predict preference of user 2211 to item 110
    print("prediction: ", lightgcn.predict(user=2211, item=110))
    # recommend 7 items for user 2211
    print("recommendation: ", lightgcn.recommend_user(user=2211, n_rec=7))

    # cold-start prediction
    print(
        "cold prediction: ",
        lightgcn.predict(user="ccc", item="not item", cold_start="average"),
    )
    # cold-start recommendation
    print(
        "cold recommendation: ",
        lightgcn.recommend_user(user="are we good?", n_rec=7, cold_start="popular"),
    )
