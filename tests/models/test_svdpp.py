import sys

import pytest
import tensorflow as tf

from libreco.algorithms import SVDpp
from libreco.data import DatasetPure
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import SAVE_PATH, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "focal", "random", None),
        ("rating", "focal", None, True),
        ("rating", "focal", "random", True),
        ("ranking", "cross_entropy", "random", False),
        ("ranking", "focal", "random", False),
        ("ranking", "cross_entropy", "random", True),
        ("ranking", "cross_entropy", "unconsumed", True),
        ("ranking", "focal", "popular", True),
        ("ranking", "unknown", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "reg, num_neg, recent_num, num_workers",
    [(None, 1, 10, 0), (0.001, 3, 1, 2), (2.0, 1, None, 1), (0.001, 3, 0, 0)],
)
def test_svdpp(
    pure_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    reg,
    num_neg,
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
            SVDpp(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            SVDpp(task, data_info).fit(train_data, neg_sampling)
    elif recent_num == 0:
        with pytest.raises(AssertionError):
            SVDpp(task, data_info, recent_num=recent_num).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            SVDpp(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
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
                model.fit(train_data, neg_sampling)
        else:
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
            model.save(SAVE_PATH, "svdpp_model", manual=True, inference_only=False)

            # test save and load model
            loaded_model, loaded_data_info = save_load_model(SVDpp, model, data_info)
            ptest_preds(loaded_model, task, pd_data, with_feats=False)
            ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
            with pytest.raises(RuntimeError):
                loaded_model.fit(train_data, neg_sampling)

            # test rebuild model
            train_data, new_data_info = DatasetPure.merge_trainset(
                pd_data, data_info, merge_behavior=True
            )
            if task == "ranking" and not neg_sampling:
                set_ranking_labels(train_data)
            new_model = SVDpp(
                task=task,
                data_info=new_data_info,
                embed_size=4,
                n_epochs=1,
                lr=1e-4,
                reg=reg,
                batch_size=2048,
                num_neg=num_neg,
                tf_sess_config=None,
            )
            new_model.rebuild_model(SAVE_PATH, "svdpp_model", full_assign=True)
            new_model.fit(train_data, neg_sampling)
            with pytest.raises(ValueError):
                new_model.fit(train_data, neg_sampling, eval_data=eval_data, k=10000)
