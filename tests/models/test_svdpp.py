import pytest
import tensorflow as tf

from libreco.algorithms import SVDpp
from tests.utils_data import SAVE_PATH, set_ranking_labels
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
        ("ranking", "cross_entropy", "unconsumed"),
        ("ranking", "focal", "popular"),
        ("ranking", "unknown", "popular"),
    ],
)
@pytest.mark.parametrize(
    "reg, num_neg, recent_num, num_workers",
    [(None, 1, 10, 0), (0.001, 3, 1, 2), (2.0, 1, None, 1), (0.001, 3, 0, 0)],
)
def test_svdpp(
    prepare_pure_data, task, loss_type, sampler, reg, num_neg, recent_num, num_workers
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        # train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)
        if sampler is None and loss_type == "cross_entropy":
            set_ranking_labels(train_data)

    if recent_num == 0:
        with pytest.raises(AssertionError):
            SVDpp(task, data_info, recent_num=recent_num).fit(train_data)
    elif task == "ranking" and sampler is None and loss_type == "focal":
        with pytest.raises(ValueError):
            SVDpp(task, data_info, loss_type, sampler=sampler).fit(train_data)
    else:
        model = SVDpp(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            reg=reg,
            batch_size=2048,
            sampler=sampler,
            num_neg=num_neg,
            recent_num=recent_num,
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
                num_workers=num_workers,
            )
            ptest_preds(model, task, pd_data, with_feats=False)
            ptest_recommends(model, data_info, pd_data, with_feats=False)

            # test save and load model
            loaded_model, loaded_data_info = save_load_model(SVDpp, model, data_info)
            ptest_preds(loaded_model, task, pd_data, with_feats=False)
            ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
            with pytest.raises(RuntimeError):
                loaded_model.fit(train_data)

            # test rebuild model
            model.save(SAVE_PATH, "svdpp_model", manual=True, inference_only=False)
            tf.compat.v1.reset_default_graph()
            new_model = SVDpp(
                task=task,
                data_info=data_info,
                embed_size=4,
                n_epochs=1,
                lr=1e-4,
                reg=reg,
                batch_size=2048,
                num_neg=num_neg,
                tf_sess_config=None,
            )
            new_model.rebuild_model(SAVE_PATH, "svdpp_model", full_assign=True)
            new_model.fit(train_data)
            with pytest.raises(ValueError):
                new_model.fit(train_data, eval_data=eval_data, k=10000)
