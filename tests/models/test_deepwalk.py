import pytest
import tensorflow as tf

from libreco.algorithms import DeepWalk
from tests.utils_metrics import get_metrics
from tests.utils_path import remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "norm_embed, n_walks, walk_length, window_size",
    [(True, 10, 10, 5), (False, 1, 1, 1)],
)
def test_deepwalk(
    prepare_pure_data, task, norm_embed, n_walks, walk_length, window_size
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    if task == "rating":
        with pytest.raises(AssertionError):
            _ = DeepWalk(task, data_info)
    else:
        model = DeepWalk(
            task=task,
            data_info=data_info,
            embed_size=16,
            n_epochs=2,
            norm_embed=norm_embed,
            window_size=window_size,
            n_walks=n_walks,
            walk_length=walk_length,
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
        loaded_model, loaded_data_info = save_load_model(DeepWalk, model, data_info)
        with pytest.raises(AssertionError):
            loaded_model.fit(train_data)
        with pytest.raises(AssertionError):
            loaded_model.rebuild_model(path="deepwalk_path", model_name="deepwalk2")
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "deepwalk2")
        remove_path("not_existed_path")
