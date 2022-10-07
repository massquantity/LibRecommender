import pytest

from libreco.algorithms import UserCF

from tests.utils_metrics import get_metrics
from tests.utils_path import remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("sim_type", ["cosine", "pearson", "jaccard", "unknown"])
@pytest.mark.parametrize("store_top_k", [True, False])
def test_user_cf(prepare_pure_data, task, sim_type, store_top_k):
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    model = UserCF(
        task=task,
        data_info=data_info,
        sim_type=sim_type,
        k_sim=20,
        k=10,
        eval_user_num=200,
    )

    if sim_type == "unknown":
        with pytest.raises(ValueError):
            model.fit(train_data)
    else:
        model.fit(
            train_data,
            verbose=2,
            eval_data=eval_data,
            metrics=get_metrics(task),
            store_top_k=store_top_k,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)
        model.recommend_user(1, 10, random_rec=True)
        with pytest.raises(ValueError):
            model.predict(user="cold user1", item="cold item2", cold_start="other")

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(UserCF, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "user_cf2")
        remove_path("not_existed_path")

        with pytest.raises(NotImplementedError):
            model.rebuild_model("model_path", "user_cf")
