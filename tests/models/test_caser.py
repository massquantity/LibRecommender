import pytest
import tensorflow as tf

from libreco.algorithms import Caser

# noinspection PyUnresolvedReferences
from tests.utils_data import prepare_pure_data
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
    "lr_decay, reg, num_neg, use_bn, dropout_rate, "
    "nh_filters, nv_filters, random_num",
    [(False, None, 1, False, None, 2, 4, 10), (True, 0.001, 3, True, 0.5, 4, 8, None)],
)
def test_caser(
    prepare_pure_data,
    task,
    loss_type,
    lr_decay,
    reg,
    num_neg,
    use_bn,
    dropout_rate,
    nh_filters,
    nv_filters,
    random_num,
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(
            data_info, item_gen_mode="random", num_neg=1, seed=2022
        )
        eval_data.build_negative_samples(
            data_info, item_gen_mode="random", num_neg=1, seed=2222
        )

    if task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            _ = Caser(task, data_info, loss_type)
    else:
        model = Caser(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=16,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=2048,
            num_neg=num_neg,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            nh_filters=nh_filters,
            nv_filters=nv_filters,
            recent_num=None,
            random_num=random_num,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(Caser, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
