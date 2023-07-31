import pytest
import tensorflow as tf

from libreco.algorithms import Transformer
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_metrics import get_metrics
from tests.utils_multi_sparse_models import fit_multi_sparse
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type",
    [
        ("rating", "whatever"),
        ("ranking", "cross_entropy"),
        ("ranking", "unknown"),
    ],
)
@pytest.mark.parametrize(
    "lr_decay, reg, use_bn, num_heads, num_tfm_layers, use_causal_mask, feat_agg_mode",
    [
        (False, None, True, 1, 1, False, "concat"),
        (True, 0.001, False, 2, 2, True, "elementwise"),
        (True, None, False, 11, 1, False, "concat"),
        (True, None, False, 1, 1, False, "whatever"),
    ],
)
def test_transformer(
    feat_data_small,
    task,
    loss_type,
    lr_decay,
    reg,
    use_bn,
    num_heads,
    num_tfm_layers,
    use_causal_mask,
    feat_agg_mode,
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small

    neg_sampling = True if task == "ranking" else False
    if task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            _ = Transformer(task, data_info, loss_type)
    elif feat_agg_mode == "whatever":
        with pytest.raises(ValueError):
            _ = Transformer(task, data_info, loss_type, feat_agg_mode=feat_agg_mode)
    elif num_heads == 11:
        with pytest.raises(AssertionError):
            Transformer(task, data_info, loss_type, num_heads=num_heads).fit(
                train_data, neg_sampling
            )
    else:
        model = Transformer(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=100,
            use_bn=use_bn,
            dropout_rate=None,
            num_heads=num_heads,
            num_tfm_layers=num_tfm_layers,
            use_causal_mask=use_causal_mask,
            feat_agg_mode=feat_agg_mode,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            neg_sampling=neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=40,
            num_workers=0,
        )
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=True)
        ptest_recommends(model, data_info, pd_data, with_feats=True)
        ptest_dyn_recommends(model, pd_data)


def test_transformer_multi_sparse(multi_sparse_data_small):
    task = "ranking"
    pd_data, train_data, eval_data, data_info = multi_sparse_data_small
    model = fit_multi_sparse(Transformer, train_data, eval_data, data_info)
    ptest_preds(model, task, pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(Transformer, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=True)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
    with pytest.raises(RuntimeError):
        loaded_model.fit(train_data, neg_sampling=True)
