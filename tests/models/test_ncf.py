import pytest
import tensorflow as tf

from libreco.algorithms import NCF

from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type",
    [
        ("rating", "whatever"),
        ("ranking", "cross_entropy"),
        ("ranking", "focal"),
        ("ranking", "unknown"),
    ],
)
@pytest.mark.parametrize(
    "lr_decay, reg, num_neg, use_bn, dropout_rate, hidden_units",
    [(False, None, 1, False, None, "128,64,32"), (True, 0.001, 3, True, 0.5, "1,1,1")],
)
def test_ncf(
    prepare_pure_data,
    task,
    loss_type,
    lr_decay,
    reg,
    num_neg,
    use_bn,
    dropout_rate,
    hidden_units,
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    if task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            _ = NCF(task, data_info, loss_type)
    else:
        model = NCF(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=16,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=1024,
            num_neg=num_neg,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=200,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(NCF, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
