import pytest
import tensorflow as tf

from libreco.algorithms import GraphSageDGL
from tests.utils_data import remove_path
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "paradigm, aggregator_type, loss_type, sampler, num_neg",
    [
        ("whatever", "mean", "cross_entropy", "random", 1),
        ("u2i", "mean", "cross_entropy", "random", 1),
        ("i2i", "gcn", "cross_entropy", "random", 1),
        ("u2i", "gcn", "cross_entropy", None, 0),
        ("u2i", "gcn", "cross_entropy", "random", 0),
        ("u2i", "pool", "focal", None, 1),
        ("u2i", "pool", "focal", "unconsumed", 3),
        ("u2i", "lstm", "focal", "popular", 2),
        ("i2i", "mean", "focal", "random", 3),
        ("u2i", "pool", "bpr", "popular", 2),
        ("u2i", "lstm", "bpr", "unconsumed", 2),
        ("u2i", "whatever", "bpr", "random", 2),
        ("i2i", "gcn", "max_margin", "random", 2),
        ("i2i", "pool", "max_margin", "popular", 1),
        ("i2i", "lstm", "max_margin", None, 2),
        ("u2i", "lstm", "whatever", "random", 1),
        ("u2i", "gcn", "bpr", "whatever", 1),
        ("i2i", "mean", "bpr", "out-batch", 1),
        ("i2i", "mean", "bpr", "unconsumed", 5),
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
            0.0,
            "unpopular",
            True,
            0,
        ),
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
            2,
        ),
    ],
)
def test_pinsage(
    prepare_feat_data,
    task,
    paradigm,
    aggregator_type,
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
    num_workers,
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
        "aggregator_type": aggregator_type,
        "sampler": sampler,
        "num_neg": num_neg,
    }

    if task == "rating":
        with pytest.raises(ValueError):
            _ = GraphSageDGL(**params)
    elif paradigm == "whatever":
        with pytest.raises(ValueError):
            _ = GraphSageDGL(**params)
    elif aggregator_type == "whatever":
        with pytest.raises(ValueError):
            _ = GraphSageDGL(**params)
    elif loss_type == "whatever":
        with pytest.raises(ValueError):
            _ = GraphSageDGL(**params)
    elif not sampler or sampler == "whatever":
        with pytest.raises(ValueError):
            GraphSageDGL(**params).fit(train_data)
    elif paradigm == "i2i" and sampler == "unconsumed":
        with pytest.raises(ValueError):
            GraphSageDGL(**params).fit(train_data)
    elif loss_type == "cross_entropy" and sampler and num_neg <= 0:
        with pytest.raises(AssertionError):
            GraphSageDGL(**params).fit(train_data)
    elif loss_type == "max_margin" and not sampler:
        with pytest.raises(ValueError):
            GraphSageDGL(**params).fit(train_data)
    elif num_workers != 0:
        with pytest.raises(AssertionError):
            GraphSageDGL(**params).fit(train_data, num_workers=num_workers)
    else:
        model = GraphSageDGL(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            paradigm=paradigm,
            aggregator_type=aggregator_type,
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
            num_workers=num_workers,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(GraphSageDGL, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        with pytest.raises(RuntimeError):
            loaded_model.fit(train_data)
        model.save("not_existed_path", "graphsage2")
        remove_path("not_existed_path")
