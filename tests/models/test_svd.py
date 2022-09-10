import pytest
import tensorflow as tf

from libreco.algorithms import SVD

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
@pytest.mark.parametrize("reg, num_neg", [(None, 1), (0.001, 3)])
def test_svd(prepare_pure_data, task, loss_type, reg, num_neg):
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
            _ = SVD(task, data_info, loss_type)
    else:
        model = SVD(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=16,
            n_epochs=2,
            lr=1e-4,
            reg=reg,
            batch_size=256,
            num_neg=num_neg,
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

        # test save and load model manual
        loaded_model, loaded_data_info = save_load_model(SVD, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
