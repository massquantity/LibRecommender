import pandas as pd

from libreco.algorithms import DeepFM
from libreco.data import DatasetFeat, split_by_ratio_chrono, split_multi_value

pd.set_option("display.max_columns", 20)


if __name__ == "__main__":
    # choose data named "sample_movielens_genre.csv"
    data = pd.read_csv("sample_data/sample_movielens_genre.csv", header=0)
    print("=" * 30, "original data", "=" * 30)
    print(data.head(), "\n")

    sparse_col = ["sex", "occupation"]
    dense_col = ["age"]
    multi_value_col = ["genre"]  # specify multi-value feature
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre"]

    # The "max_len" parameter means the maximum number of sub-features after transformation.
    # If it is None, will use max category length a sample can have across the whole data.
    # Note if not None, it should be a list, because there are possibly many `multi_value` features.
    data, multi_sparse_col, multi_user_col, multi_item_col = split_multi_value(
        data,
        multi_value_col,
        sep="|",
        max_len=[3],
        pad_val="missing",
        user_col=user_col,
        item_col=item_col,
    )

    print("multi_sparse_col: ", multi_sparse_col)
    print("multi_user_col: ", multi_user_col)
    print("multi_item_col: ", multi_item_col)

    # the multi-value feature may belong to user or item, so we add them together.
    user_col += multi_user_col
    item_col += multi_item_col
    # we do not need the original genre feature anymore
    item_col.remove("genre")
    print("final user col: ", user_col)
    print("final item col: ", item_col, "\n")
    print("=" * 30, "transformed data", "=" * 30)
    print(data.head(), "\n")

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)

    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        user_col=user_col,
        item_col=item_col,
        sparse_col=sparse_col,
        dense_col=dense_col,
        multi_sparse_col=multi_sparse_col,
        pad_val=["missing"],  # specify padding value
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    deepfm = DeepFM(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
        multi_sparse_combiner="normal",  # specify multi_sparse combiner
    )
    deepfm.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
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
    print("prediction: ", deepfm.predict(user=1, item=2333))
    print("recommendation: ", deepfm.recommend_user(user=1, n_rec=7))
