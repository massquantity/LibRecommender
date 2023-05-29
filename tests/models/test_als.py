import pytest

from libreco.algorithms import ALS
from libreco.algorithms.als import least_squares, least_squares_cg
from libreco.evaluation import evaluate
from tests.utils_data import SAVE_PATH, remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "neg_sampling, reg, alpha",
    [
        (True, 1, 5),
        (True, -1.0, 5),
        (True, None, 5),
        (True, 0, 5),
        (True, 0.001, 10),
        (True, 0.1, 100),
        (False, 0.1, 100),
        (False, 1.1, 100),
        (None, 0.1, 100),
    ],
)
def test_als(pure_data_small, task, neg_sampling, reg, alpha):
    pd_data, train_data, eval_data, data_info = pure_data_small
    if task == "ranking" and reg == 1.1:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if reg in (1, -1.0, None, 0):
        with pytest.raises(ValueError):
            ALS(task, data_info, reg=reg)
    elif neg_sampling is None:
        with pytest.raises(AssertionError):
            ALS(task, data_info, reg=reg).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            ALS(task, data_info).fit(train_data, neg_sampling)
    elif task == "ranking" and not neg_sampling and reg == 0.1:
        with pytest.raises(ValueError):
            ALS(task, data_info, reg=reg).fit(train_data, neg_sampling)
    else:
        model = ALS(
            task=task,
            data_info=data_info,
            embed_size=16,
            n_epochs=2,
            reg=reg,
            alpha=alpha,
            lower_upper_bound=[1, 5],
        )
        model.fit(
            train_data,
            neg_sampling=neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        evaluate(
            model,
            eval_data,
            neg_sampling=neg_sampling,
            eval_batch_size=8192,
            k=10,
            metrics=get_metrics(task),
        )

        with pytest.raises(ValueError):
            evaluate(model, eval_data, neg_sampling=True, metrics="whatever")
        with pytest.raises(TypeError):
            evaluate(model, eval_data, neg_sampling=True, k=1.0)

        # test save and load model
        if task == "ranking" and reg == 0.1:
            loaded_model, loaded_data_info = save_load_model(ALS, model, data_info)
            ptest_preds(loaded_model, task, pd_data, with_feats=False)
            ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
            loaded_model.rebuild_model(SAVE_PATH, "als_model")
            model.save("not_existed_path", "als2")
            remove_path("not_existed_path")

        # test optimize functions
        with pytest.raises(ValueError):
            least_squares(
                train_data.sparse_interaction,
                X=model.user_embeds_np,
                Y=model.item_embeds_np,
                reg=5.0,
                embed_size=16,
                num=model.n_users,
                mode="whatever",
            )
        with pytest.raises(ValueError):
            least_squares_cg(
                train_data.sparse_interaction,
                X=model.user_embeds_np,
                Y=model.item_embeds_np,
                reg=5.0,
                embed_size=16,
                num=model.n_users,
                mode="whatever",
                cg_steps=3,
            )

        if task == "rating":
            least_squares(
                train_data.sparse_interaction,
                X=model.user_embeds_np,
                Y=model.item_embeds_np,
                reg=5.0,
                embed_size=16,
                num=model.n_users,
                mode="explicit",
            )
            least_squares_cg(
                train_data.sparse_interaction,
                X=model.user_embeds_np,
                Y=model.item_embeds_np,
                reg=5.0,
                embed_size=16,
                num=model.n_users,
                mode="explicit",
                cg_steps=3,
            )

        if task == "ranking":
            least_squares(
                train_data.sparse_interaction.T.tocsr(),
                X=model.item_embeds_np,
                Y=model.user_embeds_np,
                reg=5.0,
                embed_size=16,
                num=model.n_items,
                mode="implicit",
            )
            least_squares_cg(
                train_data.sparse_interaction.T.tocsr(),
                X=model.item_embeds_np,
                Y=model.user_embeds_np,
                reg=5.0,
                embed_size=16,
                num=model.n_items,
                mode="implicit",
                cg_steps=3,
            )


# def test_failed_import(monkeypatch):
#    with monkeypatch.context() as m:
#        m.delitem(sys.modules, "libreco.algorithms.als")
#        m.setitem(sys.modules, "libreco.algorithms._als", None)
#        with pytest.raises((ImportError, ModuleNotFoundError)):
#            from libreco.algorithms.als import ALS
