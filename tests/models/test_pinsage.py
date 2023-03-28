import sys

import pytest
import tensorflow as tf

from libreco.algorithms import PinSage
from tests.utils_data import remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "paradigm, loss_type, sampler, num_neg, neg_sampling",
    [
        ("whatever", "cross_entropy", "random", 1, True),
        ("u2i", "cross_entropy", "random", 1, False),
        ("u2i", "focal", "random", 1, False),
        ("u2i", "cross_entropy", "random", 1, True),
        ("i2i", "cross_entropy", "random", 1, True),
        ("u2i", "cross_entropy", "random", 0, True),
        ("u2i", "focal", "unconsumed", 3, True),
        ("u2i", "focal", "popular", 2, True),
        ("i2i", "focal", "random", 3, True),
        ("u2i", "bpr", "popular", 2, True),
        ("u2i", "bpr", "unconsumed", 2, True),
        ("u2i", "bpr", "random", 2, True),
        ("i2i", "max_margin", "random", 2, True),
        ("i2i", "max_margin", "popular", 1, True),
        ("u2i", "whatever", "random", 1, True),
        ("u2i", "bpr", "whatever", 1, True),
        ("i2i", "bpr", "out-batch", 1, True),
        ("i2i", "bpr", "unconsumed", 5, True),
    ],
)
@pytest.mark.parametrize(
    "reg,"
    "dropout_rate,"
    "lr_decay,"
    "epsilon,"
    "amsgrad,"
    "remove_edges,"
    "num_layers,"
    "num_neighbors,"
    "num_walks,"
    "neighbor_walk_len,"
    "sample_walk_len,"
    "termination_prob,"
    "margin,"
    "start_node,"
    "focus_start,"
    "num_workers",
    [
        (
            0.0,
            0.0,
            False,
            1e-8,
            False,
            True,
            2,
            3,
            3,
            2,
            2,
            0.5,
            1.0,
            "random",
            False,
            0,
        ),
        (
            0.01,
            0.2,
            True,
            4e-5,
            True,
            False,
            3,
            1,
            1,
            1,
            1,
            0.0,
            0.0,
            "unpopular",
            True,
            2,
        ),
    ],
)
def test_pinsage(
    feat_data_small,
    task,
    paradigm,
    loss_type,
    sampler,
    num_neg,
    neg_sampling,
    reg,
    dropout_rate,
    lr_decay,
    epsilon,
    amsgrad,
    remove_edges,
    num_layers,
    num_neighbors,
    num_walks,
    neighbor_walk_len,
    sample_walk_len,
    termination_prob,
    margin,
    start_node,
    focus_start,
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

    params = {
        "task": task,
        "data_info": data_info,
        "loss_type": loss_type,
        "paradigm": paradigm,
        "sampler": sampler,
        "num_neg": num_neg,
    }

    if task == "rating":
        with pytest.raises(ValueError):
            _ = PinSage(**params)
    elif paradigm == "whatever":
        with pytest.raises(ValueError):
            _ = PinSage(**params)
    elif loss_type == "whatever":
        with pytest.raises(ValueError):
            _ = PinSage(**params)
    elif loss_type == "focal" and neg_sampling is False:
        with pytest.raises(ValueError):
            PinSage(**params).fit(train_data, neg_sampling)
    elif paradigm == "i2i" and sampler == "unconsumed":
        with pytest.raises(ValueError):
            PinSage(**params).fit(train_data, neg_sampling)
    elif loss_type == "cross_entropy" and neg_sampling and num_neg <= 0:
        with pytest.raises(AssertionError):
            PinSage(**params).fit(train_data, neg_sampling)
    elif not sampler or sampler == "whatever":
        with pytest.raises(ValueError):
            PinSage(**params).fit(train_data, neg_sampling)
    else:
        model = PinSage(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            paradigm=paradigm,
            embed_size=4,
            n_epochs=1,
            lr=3e-4,
            lr_decay=lr_decay,
            epsilon=epsilon,
            amsgrad=amsgrad,
            batch_size=80,
            reg=reg,
            dropout_rate=dropout_rate,
            num_neg=num_neg,
            sampler=sampler,
            remove_edges=remove_edges,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            num_walks=num_walks,
            neighbor_walk_len=neighbor_walk_len,
            sample_walk_len=sample_walk_len,
            termination_prob=termination_prob,
            margin=margin,
            start_node=start_node,
            focus_start=focus_start,
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
        loaded_model, loaded_data_info = save_load_model(PinSage, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "pinsage2")
        remove_path("not_existed_path")
