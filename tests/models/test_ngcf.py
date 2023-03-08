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
    "loss_type, sampler, num_neg",
    [
        ("cross_entropy", "random", 1),
        ("cross_entropy", "random", 0),
        ("cross_entropy", None, 1),
        ("focal", None, 1),
        ("focal", "unconsumed", 3),
        ("bpr", "popular", 3),
        ("max_margin", "random", 2),
        ("max_margin", None, 2),
        ("whatever", "random", 1),
        ("bpr", "whatever", 1),
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
    prepare_pure_data,
    task,
    loss_type,
    sampler,
    num_neg,
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
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        # train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)
        if sampler is None and loss_type == "cross_entropy":
            set_ranking_labels(train_data)

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
    elif loss_type == "cross_entropy" and sampler and num_neg <= 0:
        with pytest.raises(AssertionError):
            NGCF(**params).fit(train_data)
    elif loss_type == "max_margin" and not sampler:
        with pytest.raises(ValueError):
            NGCF(**params).fit(train_data)
    elif task == "ranking" and sampler is None and loss_type == "focal":
        with pytest.raises(ValueError):
            NGCF(**params).fit(train_data)
    elif sampler and sampler == "whatever":
        with pytest.raises(ValueError):
            NGCF(**params).fit(train_data)
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
            batch_size=8192,
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
