import pandas as pd

from libreco.data import random_split, DatasetPure
from libreco.algorithms import SVDpp  # pure data, algorithm SVD++
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

    svdpp = SVDpp(
        task="rating",
        data_info=data_info,
        embed_size=16,
        n_epochs=3,
        lr=0.001,
        reg=None,
        batch_size=256,
    )
    # monitor metrics on eval_data during training
    svdpp.fit(train_data, verbose=2, eval_data=eval_data, metrics=["rmse", "mae", "r2"])

    # do final evaluation on test data
    print(
        "evaluate_result: ",
        evaluate(model=svdpp, data=test_data, metrics=["rmse", "mae"]),
    )
    # predict preference of user 1 to item 2333
    print("prediction: ", svdpp.predict(user=2211, item=110))
    # recommend 7 items for user 1
    print("recommendation: ", svdpp.recommend_user(user=2211, n_rec=7))

    # cold-start prediction
    print(
        "cold prediction: ",
        svdpp.predict(user="ccc", item="not item", cold_start="average"),
    )
    # cold-start recommendation
    print(
        "cold recommendation: ",
        svdpp.recommend_user(user="are we good?", n_rec=7, cold_start="popular"),
    )
