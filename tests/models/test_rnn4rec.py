import sys

import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from libreco.algorithms import RNN4Rec
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "focal", "random", None),
        ("rating", "focal", None, True),
        ("rating", "focal", "random", True),
        # ("ranking", "cross_entropy", "random", False),
        ("ranking", "focal", "random", False),
        ("ranking", "cross_entropy", "random", True),
        # ("ranking", "cross_entropy", "unconsumed", True),
        # ("ranking", "focal", "popular", True),
        ("ranking", "bpr", "unconsumed", True),
        ("ranking", "unknown", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "norm_embed, rnn_type, lr_decay, reg, num_neg, dropout_rate, "
    "hidden_units, use_layer_norm, recent_num, num_workers",
    [
        (True, "lstm", False, None, 1, None, 1, False, 10, 0),
        (False, "gru", True, 0.001, 1, 0.5, (2, 3, 1), True, 6, 2),
    ],
)
def test_rnn4rec(
    pure_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    norm_embed,
    rnn_type,
    lr_decay,
    reg,
    num_neg,
    dropout_rate,
    hidden_units,
    use_layer_norm,
    recent_num,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = pure_data_small
    if task == "ranking" and neg_sampling is False and loss_type == "cross_entropy":
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if neg_sampling is None:
        with pytest.raises(AssertionError):
            RNN4Rec(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            RNN4Rec(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            RNN4Rec(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    elif task == "ranking" and loss_type not in ("cross_entropy", "bpr", "focal"):
        with pytest.raises(ValueError):
            RNN4Rec(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = RNN4Rec(
            task=task,
            data_info=data_info,
            rnn_type=rnn_type,
            loss_type=loss_type,
            embed_size=4,
            norm_embed=norm_embed,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=100,
            sampler=sampler,
            num_neg=num_neg,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            use_layer_norm=use_layer_norm,
            recent_num=recent_num,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            num_workers=num_workers,
        )
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)
        dyn_rec = ptest_dyn_recommends(model, pd_data)

        # no user embeds to set oov
        model._assign_user_oov("user_embeds_var", "embedding")
        model._assign_user_oov("user_embeds_fake", "embedding")

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(RNN4Rec, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        loaded_dyn_rec = ptest_dyn_recommends(loaded_model, pd_data)
        assert_array_equal(dyn_rec, loaded_dyn_rec)
        with pytest.raises(RuntimeError):
            loaded_model.fit(train_data, neg_sampling)
