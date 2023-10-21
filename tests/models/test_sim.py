import sys

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from libreco.algorithms import SIM
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_multi_sparse_models import fit_multi_sparse
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "focal", "random", None),
        ("rating", "focal", None, True),
        ("rating", "focal", "random", True),
        ("ranking", "cross_entropy", "random", False),
        ("ranking", "focal", "unconsumed", False),
        ("ranking", "cross_entropy", "unconsumed", True),
        ("ranking", "focal", "popular", True),
        ("ranking", "unknown", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "lr_decay, reg, num_neg, use_bn, dropout_rate, hidden_units, "
    "search_topk, long_max_len, short_max_len, num_workers",
    [
        (False, None, 1, False, None, (64, 32), 10, 20, 10, 0),
        (True, 0.001, 3, True, 0.5, 1, 1, 1, 1, 2),
    ],
)
def test_sim(
    feat_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    lr_decay,
    reg,
    num_neg,
    use_bn,
    dropout_rate,
    hidden_units,
    search_topk,
    long_max_len,
    short_max_len,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small
    if task == "ranking" and neg_sampling is False and loss_type == "cross_entropy":
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if neg_sampling is None:  # `neg_sampling` must be True or False
        with pytest.raises(AssertionError):
            SIM(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            SIM(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            SIM(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    elif task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            SIM(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = SIM(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=100,
            sampler=sampler,
            num_neg=num_neg,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            search_topk=search_topk,
            long_max_len=long_max_len,
            short_max_len=short_max_len,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=40,
            num_workers=num_workers,
        )
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=True)
        ptest_recommends(model, data_info, pd_data, with_feats=True)
        ptest_dyn_recommends(model, pd_data)
        long_short_seq_ptest(model, pd_data)


def test_sim_multi_sparse(multi_sparse_data_small):
    task = "ranking"
    pd_data, train_data, eval_data, data_info = multi_sparse_data_small
    model = fit_multi_sparse(SIM, train_data, eval_data, data_info)
    ptest_preds(model, task, pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
    dyn_rec = ptest_dyn_recommends(model, pd_data)
    long_short_seq_ptest(model, pd_data)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(SIM, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=True)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
    loaded_dyn_rec = ptest_dyn_recommends(loaded_model, pd_data)
    assert_array_equal(dyn_rec, loaded_dyn_rec)
    with pytest.raises(RuntimeError):
        loaded_model.fit(train_data, neg_sampling=True)


def long_short_seq_ptest(model, pd_data):
    users = pd_data.user.tolist()
    user1, cold_user = users[0], -100
    rng = np.random.default_rng(42)
    long_seq = rng.integers(0, 100, size=10000)
    middle_seq = rng.integers(0, 100, size=20)
    short_seq = rng.integers(0, 100, size=8)
    little_seq = [1]

    reco1 = model.recommend_user(user=user1, n_rec=7, seq=long_seq)[user1]
    reco2 = model.recommend_user(user=user1, n_rec=7, seq=middle_seq)[user1]
    reco3 = model.recommend_user(user=user1, n_rec=7, seq=short_seq)[user1]
    reco4 = model.recommend_user(user=user1, n_rec=7, seq=little_seq)[user1]
    assert len(reco1) == len(reco2) == len(reco3) == len(reco4) == 7

    cold1 = model.recommend_user(user=cold_user, n_rec=7, seq=long_seq)[cold_user]
    cold2 = model.recommend_user(user=cold_user, n_rec=7, seq=middle_seq)[cold_user]
    cold3 = model.recommend_user(user=cold_user, n_rec=7, seq=short_seq)[cold_user]
    cold4 = model.recommend_user(user=cold_user, n_rec=7, seq=little_seq)[cold_user]
    assert len(cold1) == len(cold2) == len(cold3) == len(cold4) == 7
