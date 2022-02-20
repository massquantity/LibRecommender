import os
from pathlib import Path

import pytest
import pandas as pd
import tensorflow as tf
from libreco.data import split_by_ratio_chrono, DatasetFeat

from libreco.algorithms import YouTuBeRetrieval


# According to the paper, the YouTuBeRetrieval model can not use item features.
from tests.utils_reco import recommend_in_former_consumed


@pytest.fixture
def prepare_feat_data():
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent.parent),
        "sample_data",
        "sample_movielens_merged.csv"
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation"],
        dense_col=["age"],
        user_col=["sex", "age", "occupation"],
        item_col=[]
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    return pd_data, train_data, eval_data, data_info


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("lr_decay, reg, use_bn, dropout_rate", [
    (False, None, False, None),
    (True, 0.001, True, 0.5)
])
@pytest.mark.parametrize("num_sampled_per_batch", [None, 1, 3111])
@pytest.mark.parametrize("loss_type", ["nce", "sampled_softmax"])
def test_youtube_retrieval(
        prepare_feat_data,
        task,
        lr_decay,
        reg,
        use_bn,
        dropout_rate,
        num_sampled_per_batch,
        loss_type
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_feat_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, item_gen_mode="random",
                                          num_neg=1, seed=2022)
        eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                         num_neg=1, seed=2222)
    metrics = (
        ["rmse", "mae", "r2"]
        if task == "rating"
        else ["roc_auc", "precision", "ndcg"]
    )
    model = YouTuBeRetrieval(
        task=task,
        data_info=data_info,
        embed_size=16,
        n_epochs=2,
        lr=1e-4,
        lr_decay=lr_decay,
        reg=reg,
        batch_size=256,
        use_bn=use_bn,
        dropout_rate=dropout_rate,
        num_sampled_per_batch=num_sampled_per_batch,
        loss_type=loss_type,
        tf_sess_config=None
    )

    if task == "rating":
        with pytest.raises(AssertionError):
            model.fit(
                train_data,
                verbose=2,
                shuffle=True,
                eval_data=eval_data,
                metrics=metrics
            )
    else:
        model.fit(
            train_data,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=metrics
        )
        pred = model.predict(user=1, item=2333)
        # prediction in range
        if task == "rating":
            assert 1 <= pred <= 5
        else:
            assert 0 <= pred <= 1

        cold_pred1 = model.predict(user="cold user1", item="cold item2")
        cold_pred2 = model.predict(user="cold user2", item="cold item2")
        assert cold_pred1 == cold_pred2

        # cold start strategy
        with pytest.raises(ValueError):
            model.recommend_user(user=-99999, n_rec=7, cold_start="sss")
        # different recommendation for different users
        reco_take_one = [i[0] for i in model.recommend_user(user=1, n_rec=7)]
        reco_take_two = [i[0] for i in model.recommend_user(user=2, n_rec=7)]
        assert len(reco_take_one) == len(reco_take_two) == 7
        assert reco_take_one != reco_take_two
        assert not recommend_in_former_consumed(data_info, reco_take_one, 1)
        assert not recommend_in_former_consumed(data_info, reco_take_two, 2)
        cold_reco1 = model.recommend_user(user=-99999, n_rec=3)
        cold_reco2 = model.recommend_user(user=-1, n_rec=3)
        assert cold_reco1 == cold_reco2
