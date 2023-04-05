import time

import pandas as pd

from libreco.algorithms import DIN
from libreco.data import DatasetFeat

if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv("sample_data/sample_movielens_merged.csv", sep=",", header=0)

    # specify complete columns information
    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]

    train_data, data_info = DatasetFeat.build_trainset(
        data, user_col, item_col, sparse_col, dense_col
    )

    din = DIN(
        "ranking",
        data_info,
        loss_type="focal",
        embed_size=16,
        n_epochs=1,
        lr=3e-3,
        lr_decay=False,
        reg=None,
        batch_size=64,
        sampler="popular",
        num_neg=1,
        use_bn=True,
        hidden_units=(110, 32),
        recent_num=10,
        tf_sess_config=None,
        use_tf_attention=True,
    )
    din.fit(train_data, neg_sampling=True, verbose=2, shuffle=True, eval_data=None)

    print(
        "feat recommendation: ",
        din.recommend_user(user=4617, n_rec=7, user_feats={"sex": "F", "age": 3}),
    )
    print(
        "seq recommendation1: ",
        din.recommend_user(
            user=4617,
            n_rec=7,
            seq=[4, 0, 1, 222, "cold item", 222, 12, 1213, 1197, 1193],
        ),
    )
    print(
        "seq recommendation2: ",
        din.recommend_user(
            user=4617,
            n_rec=7,
            seq=["cold item", 1270, 2161, 110, 3827, 12, 34, 1273, 1589],
        ),
    )
    print(
        "feat & seq recommendation1: ",
        din.recommend_user(
            user=1, n_rec=7, user_feats={"sex": "F", "age": 3}, seq=[4, 0, 1, 222]
        ),
    )
    print(
        "feat & seq recommendation2: ",
        din.recommend_user(
            user=1, n_rec=7, user_feats={"sex": "M", "age": 33}, seq=[4, 0, 337, 1497]
        ),
    )
