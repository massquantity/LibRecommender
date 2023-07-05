import pandas as pd

from libreco.algorithms import RNN4Rec
from libreco.data import DatasetPure, split_by_ratio_chrono
from libreco.utils.misc import colorize

try:
    import nmslib  # noqa: F401

    approximate = True
    print_str = "using `nmslib` for similarity search"
    print(f"{colorize(print_str, 'cyan')}")
except (ImportError, ModuleNotFoundError):
    approximate = False
    print_str = "failed to import `nmslib`, using `numpy` for similarity search"
    print(f"{colorize(print_str, 'cyan')}")


if __name__ == "__main__":
    data = pd.read_csv(
        "sample_data/sample_movielens_rating.dat",
        sep="::",
        names=["user", "item", "label", "time"],
    )

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)

    rnn = RNN4Rec(
        "ranking",
        data_info,
        rnn_type="lstm",
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=2,
        lr=0.001,
        lr_decay=False,
        hidden_units=16,
        reg=None,
        batch_size=2048,
        num_neg=1,
        dropout_rate=None,
        recent_num=10,
        tf_sess_config=None,
    )
    rnn.fit(train_data, neg_sampling=True, verbose=2)

    # `sim_type` should either be `cosine` or `inner-product`
    rnn.init_knn(approximate=approximate, sim_type="cosine")
    print("embedding for user 1: ", rnn.get_user_embedding(user=1))
    print("embedding for item 2: ", rnn.get_item_embedding(item=2))
    print()

    print(" 3 most similar users for user 1: ", rnn.search_knn_users(user=1, k=3))
    print(" 3 most similar items for item 2: ", rnn.search_knn_items(item=2, k=3))
    print()

    user_embed = rnn.dyn_user_embedding(user=1, seq=[0, 10, 100])
    print("generate embedding for user 1: ", user_embed)
