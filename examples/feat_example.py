import pandas as pd

from libreco.algorithms import YouTubeRanking
from libreco.data import DatasetFeat, split_by_ratio_chrono

if __name__ == "__main__":
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)

    # split into train and test data based on time
    train_data, test_data = split_by_ratio_chrono(data, test_size=0.2)

    # specify complete columns information
    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    test_data = DatasetFeat.build_testset(test_data)
    print(data_info)  # n_users: 5953, n_items: 3209, data density: 0.4213 %

    ytb_ranking = YouTubeRanking(
        task="ranking",
        data_info=data_info,
        embed_size=16,
        n_epochs=3,
        lr=1e-4,
        batch_size=512,
        use_bn=True,
        hidden_units=(128, 64, 32),
    )
    ytb_ranking.fit(
        train_data,
        neg_sampling=True,  # sample negative items train and eval data
        verbose=2,
        shuffle=True,
        eval_data=test_data,
        metrics=["loss", "roc_auc", "precision", "recall", "map", "ndcg"],
    )

    # predict preference of user 2211 to item 110
    print("prediction: ", ytb_ranking.predict(user=2211, item=110))
    # recommend 7 items for user 2211
    print("recommendation: ", ytb_ranking.recommend_user(user=2211, n_rec=7))

    # cold-start prediction
    print(
        "cold prediction: ",
        ytb_ranking.predict(user="ccc", item="not item", cold_start="average"),
    )
    # cold-start recommendation
    print(
        "cold recommendation: ",
        ytb_ranking.recommend_user(user="are we good?", n_rec=7, cold_start="popular"),
    )
