import sys

import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from libreco.algorithms import WaveNet
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
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
    "norm_embed, lr_decay, reg, num_neg, dropout_rate, use_bn, "
    "n_filters, n_blocks, n_layers_per_block, recent_num, num_workers",
    [
        (True, False, None, 1, None, False, 16, 1, 4, 10, 0),
        (False, True, 0.001, 3, 0.5, True, 32, 4, 2, 6, 2),
    ],
)
def test_wave_net(
    pure_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    norm_embed,
    lr_decay,
    reg,
    num_neg,
    dropout_rate,
    use_bn,
    n_filters,
    n_blocks,
    n_layers_per_block,
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
            WaveNet(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            WaveNet(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            WaveNet(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    elif task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            WaveNet(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = WaveNet(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            norm_embed=norm_embed,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=40,
            sampler=sampler,
            num_neg=num_neg,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            n_filters=n_filters,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            recent_num=recent_num,
            tf_sess_config=None,
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
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)
        dyn_rec = ptest_dyn_recommends(model, pd_data)

        model._assign_user_oov("user_embeds_var", "embedding")
        with pytest.raises(ValueError):
            model._assign_user_oov("user_embeds_fake", "embedding")

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(WaveNet, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        loaded_dyn_rec = ptest_dyn_recommends(loaded_model, pd_data)
        assert_array_equal(dyn_rec, loaded_dyn_rec)
