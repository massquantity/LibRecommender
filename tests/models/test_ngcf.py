import sys

import pytest
import tensorflow as tf

from libreco.algorithms import NGCF
from tests.utils_data import remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "loss_type, sampler, num_neg, neg_sampling",
    [
        ("focal", None, 1, True),
        ("focal", "random", 1, None),
        ("cross_entropy", "random", 1, True),
        ("cross_entropy", "random", 0, True),
        ("cross_entropy", "random", 1, False),
        ("focal", "unconsumed", 1, False),
        ("focal", "unconsumed", 3, True),
        ("focal", "popular", 3, True),
        ("bpr", "popular", 3, True),
        ("max_margin", "random", 2, True),
        ("max_margin", None, 2, False),
        ("whatever", "random", 1, True),
    ],
)
@pytest.mark.parametrize(
    "reg, node_dropout, message_dropout, lr_decay, epsilon, amsgrad, hidden_units, device, num_workers",
    [
        (0.0, 0.0, 0.0, False, 1e-8, False, 1, "cpu", 0),
        (0.01, 0.2, 0.2, True, 4e-5, True, (16, 16), "cuda", 2),
    ],
)
def test_ngcf(
    pure_data_small,
    task,
    loss_type,
    sampler,
    num_neg,
    neg_sampling,
    reg,
    node_dropout,
    message_dropout,
    lr_decay,
    epsilon,
    amsgrad,
    hidden_units,
    device,
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

    params = {
        "task": task,
        "data_info": data_info,
        "loss_type": loss_type,
        "sampler": sampler,
        "num_neg": num_neg,
    }

    if task == "rating":
        with pytest.raises(ValueError):
            _ = NGCF(**params)
    elif loss_type == "whatever":
        with pytest.raises(ValueError):
            _ = NGCF(**params)
    elif neg_sampling is None:
        with pytest.raises(AssertionError):
            NGCF(**params).fit(train_data, neg_sampling)
    elif loss_type != "cross_entropy" and (not neg_sampling or sampler is None):
        with pytest.raises(ValueError):
            NGCF(**params).fit(train_data, neg_sampling)
    elif loss_type == "cross_entropy" and neg_sampling and num_neg <= 0:
        with pytest.raises(AssertionError):
            NGCF(**params).fit(train_data, neg_sampling)
    else:
        model = NGCF(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            epsilon=epsilon,
            amsgrad=amsgrad,
            batch_size=40,
            reg=reg,
            node_dropout=node_dropout,
            message_dropout=message_dropout,
            num_neg=num_neg,
            sampler=sampler,
            hidden_units=hidden_units,
            device=device,
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
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(NGCF, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "ngcf2")
        remove_path("not_existed_path")
