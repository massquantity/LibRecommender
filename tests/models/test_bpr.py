import sys

import pytest
import tensorflow as tf

from libreco.algorithms import BPR
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import remove_path
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "whatever", "random", True),
        ("ranking", "cross_entropy", None, True),
        ("ranking", "cross_entropy", "random", True),
        ("ranking", "focal", "unconsumed", True),
        ("ranking", "bpr", None, True),
        ("ranking", "bpr", "random", False),
        ("ranking", "bpr", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "norm_embed, reg, num_neg, use_tf, optimizer, num_workers",
    [
        (True, None, 1, False, "unknown", 0),
        (True, None, 1, True, "sgd", 0),
        (False, 0.003, 3, True, "sgd", 2),
        (False, 0.001, 3, False, "sgd", 0),
        (False, None, 1, False, "momentum", 0),
        (False, 0.001, 3, False, "adam", 0),
    ],
)
def test_bpr(
    pure_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    norm_embed,
    reg,
    num_neg,
    use_tf,
    optimizer,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = pure_data_small

    if task == "rating" or (task == "ranking" and loss_type != "bpr"):
        with pytest.raises(AssertionError):
            _ = BPR(task, data_info, loss_type)
    elif sampler is None or not neg_sampling:
        with pytest.raises(ValueError):
            BPR(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    else:
        model = BPR(
            task=task,
            data_info=data_info,
            embed_size=16,
            norm_embed=norm_embed,
            n_epochs=2,
            lr=1e-4,
            reg=reg,
            batch_size=40,
            sampler=sampler,
            num_neg=num_neg,
            use_tf=use_tf,
            optimizer=optimizer,
        )
        if optimizer == "unknown":
            with pytest.raises(ValueError):
                model.fit(
                    train_data,
                    neg_sampling,
                    verbose=2,
                    shuffle=True,
                    eval_data=eval_data,
                    metrics=get_metrics(task),
                )
        else:
            model.fit(
                train_data,
                neg_sampling,
                verbose=2,
                shuffle=True,
                eval_data=eval_data,
                metrics=get_metrics(task),
                eval_user_num=200,
                num_workers=num_workers,
            )
            if use_tf:
                ptest_tf_variables(model)
            ptest_preds(model, task, pd_data, with_feats=False)
            ptest_recommends(model, data_info, pd_data, with_feats=False)
            with pytest.raises(ValueError):
                model.fit(train_data, neg_sampling, eval_data=eval_data, k=10000)

            # test save and load model
            loaded_model, loaded_data_info = save_load_model(BPR, model, data_info)
            ptest_preds(loaded_model, task, pd_data, with_feats=False)
            ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
            with pytest.raises(RuntimeError):
                loaded_model.fit(train_data, neg_sampling)
            model.save("not_existed_path", "bpr2")
            remove_path("not_existed_path")


# def test_failed_import(monkeypatch):
#    with monkeypatch.context() as m:
#        m.delitem(sys.modules, "libreco.algorithms.bpr")
#        m.setitem(sys.modules, "libreco.algorithms._bpr", None)
#        from libreco.algorithms.bpr import BPR
