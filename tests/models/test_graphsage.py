import pytest
import tensorflow as tf

from libreco.algorithms import GraphSage
from tests.utils_metrics import get_metrics
from tests.utils_path import remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "paradigm, loss_type, sampler, num_neg",
    [
        ("whatever", "cross_entropy", "random", 1),
        ("u2i", "cross_entropy", "random", 1),
        ("i2i", "cross_entropy", "random", 1),
        ("u2i", "cross_entropy", None, 0),
        ("u2i", "cross_entropy", "random", 0),
        ("u2i", "focal", None, 1),
        ("u2i", "focal", "unconsumed", 3),
        ("u2i", "focal", "popular", 2),
        ("i2i", "focal", "random", 3),
        ("u2i", "bpr", "popular", 2),
        ("u2i", "bpr", "unconsumed", 2),
        ("u2i", "bpr", "random", 2),
        ("i2i", "max_margin", "random", 2),
        ("i2i", "max_margin", "popular", 1),
        ("i2i", "max_margin", None, 2),
        ("u2i", "whatever", "random", 1),
        ("u2i", "bpr", "whatever", 1),
        ("i2i", "bpr", "out-batch", 1),
        ("i2i", "bpr", "unconsumed", 5),
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
    "sample_walk_len,"
    "margin,"
    "start_node,"
    "focus_start",
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
            1.0,
            "random",
            False,
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
            0.0,
            "unpopular",
            True,
        ),
    ],
)
def test_graphsage(
    prepare_feat_data,
    task,
    paradigm,
    loss_type,
    sampler,
    num_neg,
    reg,
    dropout_rate,
    lr_decay,
    epsilon,
    amsgrad,
    remove_edges,
    num_layers,
    num_neighbors,
    num_walks,
    sample_walk_len,
    margin,
    start_node,
    focus_start,
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_feat_data
    if task == "ranking":
        # train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

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
            _ = GraphSage(**params)
    elif paradigm == "whatever":
        with pytest.raises(ValueError):
            _ = GraphSage(**params)
    elif loss_type == "whatever":
        with pytest.raises(ValueError):
            _ = GraphSage(**params)
    elif paradigm == "i2i" and sampler == "unconsumed":
        with pytest.raises(ValueError):
            GraphSage(**params).fit(train_data)
    elif loss_type == "cross_entropy" and sampler and num_neg <= 0:
        with pytest.raises(AssertionError):
            GraphSage(**params).fit(train_data)
    elif loss_type == "max_margin" and not sampler:
        with pytest.raises(ValueError):
            GraphSage(**params).fit(train_data)
    elif sampler and sampler == "whatever":
        with pytest.raises(ValueError):
            GraphSage(**params).fit(train_data)
    else:
        model = GraphSage(
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
            batch_size=8192,
            reg=reg,
            dropout_rate=dropout_rate,
            num_neg=num_neg,
            sampler=sampler,
            remove_edges=remove_edges,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            num_walks=num_walks,
            sample_walk_len=sample_walk_len,
            margin=margin,
            start_node=start_node,
            focus_start=focus_start,
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
        loaded_model, loaded_data_info = save_load_model(GraphSage, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "graphsage2")
        remove_path("not_existed_path")
