import pytest
import tensorflow as tf

from libreco.algorithms import Item2Vec
from tests.utils_data import remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("norm_embed, window_size", [(True, 5), (False, None)])
@pytest.mark.parametrize("neg_sampling", [True, False, None])
def test_item2vec(pure_data_small, task, norm_embed, window_size, neg_sampling):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = pure_data_small
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if task == "rating":
        with pytest.raises(AssertionError):
            _ = Item2Vec(task, data_info)
    elif neg_sampling is None:
        with pytest.raises(AssertionError):
            Item2Vec(task, data_info).fit(train_data, neg_sampling)
    else:
        model = Item2Vec(
            task=task,
            data_info=data_info,
            embed_size=16,
            n_epochs=2,
            norm_embed=norm_embed,
            window_size=window_size,
        )
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(Item2Vec, model, data_info)
        with pytest.raises(RuntimeError):
            loaded_model.fit(train_data, neg_sampling)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "item2vec2")
        remove_path("not_existed_path")
