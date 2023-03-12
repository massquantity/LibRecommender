import pytest
import tensorflow as tf

from libreco.algorithms import RNN4Rec
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler",
    [
        ("rating", "focal", "random"),
        ("ranking", "cross_entropy", None),
        ("ranking", "focal", None),
        ("ranking", "cross_entropy", "random"),
        ("ranking", "bpr", "unconsumed"),
        ("ranking", "focal", "popular"),
        ("ranking", "unknown", "popular"),
    ],
)
@pytest.mark.parametrize(
    "rnn_type, lr_decay, reg, num_neg, dropout_rate, "
    "hidden_units, use_layer_norm, recent_num, num_workers",
    [
        ("lstm", False, None, 1, None, 1, False, 10, 0),
        ("gru", True, 0.001, 3, 0.5, (32, 16), True, 6, 2),
    ],
)
def test_rnn4rec(
    pure_data_small,
    task,
    loss_type,
    sampler,
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
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = pure_data_small
    if task == "ranking":
        # train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)
        if sampler is None and loss_type == "cross_entropy":
            set_ranking_labels(train_data)

    if task == "ranking" and loss_type not in ("cross_entropy", "bpr", "focal"):
        with pytest.raises(ValueError):
            RNN4Rec(task, data_info, loss_type).fit(train_data)
    elif task == "ranking" and sampler is None and loss_type == "focal":
        with pytest.raises(ValueError):
            RNN4Rec(task, data_info, loss_type, sampler=sampler).fit(train_data)
    else:
        model = RNN4Rec(
            task=task,
            data_info=data_info,
            rnn_type=rnn_type,
            loss_type=loss_type,
            embed_size=4,
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
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            num_workers=num_workers,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(RNN4Rec, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        with pytest.raises(RuntimeError):
            loaded_model.fit(train_data)
