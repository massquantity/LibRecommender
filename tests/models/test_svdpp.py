import pytest
import tensorflow as tf

from libreco.algorithms import SVDpp

from tests.utils_metrics import get_metrics
from tests.utils_path import SAVE_PATH
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
@pytest.mark.parametrize(
    "recent_num, random_sample_rate", [(None, None), (10, None), (None, 0.5)]
)
def test_svdpp(
    prepare_pure_data,
    task,
    loss_type,
    reg,
    num_neg,
    recent_num,
    random_sample_rate
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    model = SVDpp(
        task=task,
        data_info=data_info,
        loss_type=loss_type,
        embed_size=8,
        n_epochs=1,
        lr=1e-4,
        reg=reg,
        batch_size=2048,
        num_neg=num_neg,
        recent_num=recent_num,
        random_sample_rate=random_sample_rate,
        tf_sess_config=None,
    )
    if task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            model.fit(train_data)
    else:
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
        loaded_model, loaded_data_info = save_load_model(SVDpp, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)

        # test rebuild model
        model.save(SAVE_PATH, "svdpp_model", manual=True, inference_only=False)
        tf.compat.v1.reset_default_graph()
        new_model = SVDpp(
            task=task,
            data_info=data_info,
            embed_size=8,
            n_epochs=1,
            lr=1e-4,
            reg=reg,
            batch_size=2048,
            num_neg=num_neg,
            tf_sess_config=None,
        )
        with pytest.raises(ValueError):
            new_model.rebuild_model(SAVE_PATH, "svdpp_model", train_data=None)

        new_model.rebuild_model(
            SAVE_PATH, "svdpp_model", full_assign=True, train_data=train_data
        )
