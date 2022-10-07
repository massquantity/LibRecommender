import pytest
import tensorflow as tf

from libreco.algorithms import LightGCN

from tests.utils_metrics import get_metrics
from tests.utils_path import remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("reg, dropout, num_neg", [(0.0, 0.0, 1), (0.01, 0.2, 3)])
def test_lightgcn(prepare_pure_data, task, reg, dropout, num_neg):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    if task == "rating":
        with pytest.raises(AssertionError):
            _ = LightGCN(task, data_info)
    else:
        model = LightGCN(
            task=task,
            data_info=data_info,
            embed_size=16,
            n_epochs=1,
            lr=1e-4,
            batch_size=1024,
            n_layers=3,
            reg=reg,
            dropout=dropout,
            num_neg=num_neg,
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
        loaded_model, loaded_data_info = save_load_model(LightGCN, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "lightgcn2")
        remove_path("not_existed_path")
