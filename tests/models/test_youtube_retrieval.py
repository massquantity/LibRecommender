import sys

import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from libreco.algorithms import YouTubeRetrieval
from tests.utils_data import SAVE_PATH, remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends, ptest_seq_recommends
from tests.utils_save_load import save_load_model


# According to the paper, the YouTuBeRetrieval model can not use item features.
# def prepare_youtube_retrieval_data(multi_sparse=False):
#    data_path = (
#        Path(__file__).parents[1] / "sample_data" / "sample_movielens_merged.csv"
#    )
#    pd_data = pd.read_csv(data_path, sep=",", header=0)
#    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
#    if multi_sparse:
#        train_data, data_info = DatasetFeat.build_trainset(
#            train_data=train_data,
#            sparse_col=["sex", "occupation"],
#            multi_sparse_col=[["genre1", "genre2", "genre3"]],
#            dense_col=["age"],
#            user_col=["sex", "age", "occupation", "genre1", "genre2", "genre3"],
#            item_col=[],
#        )
#    else:
#        train_data, data_info = DatasetFeat.build_trainset(
#            train_data=train_data,
#            sparse_col=["sex", "occupation"],
#            dense_col=["age"],
#            user_col=["sex", "age", "occupation"],
#            item_col=[],
#        )
#    eval_data = DatasetFeat.build_testset(eval_data)
#    return pd_data, train_data, eval_data, data_info


@pytest.mark.parametrize(
    "config_feat_data_small",
    [
        {
            "sparse_col": ["sex", "occupation"],
            "dense_col": ["age"],
            "user_col": ["sex", "occupation", "age"],
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "norm_embed, lr_decay, reg, use_bn, dropout_rate, recent_num, random_num, hidden_units, num_workers",
    [
        (True, False, None, False, None, 10, None, 1, 0),
        (False, True, 0.001, True, 0.5, None, 10, [16, 16], 1),
        (True, True, 0.001, False, None, None, None, (4, 4, 4), 2),
        (False, False, None, False, None, 10, None, "64,64", 0),
        (False, True, 0.001, True, 0.5, None, 10, [1, 2, 4.22], 0),
    ],
)
@pytest.mark.parametrize("num_sampled_per_batch", [None, 1, 10])
@pytest.mark.parametrize("neg_sampling", ["wrong_labels", False, True])
@pytest.mark.parametrize("loss_type", ["nce", "sampled_softmax", "unknown"])
def test_youtube_retrieval(
    config_feat_data_small,
    task,
    norm_embed,
    lr_decay,
    reg,
    use_bn,
    dropout_rate,
    num_sampled_per_batch,
    neg_sampling,
    loss_type,
    recent_num,
    random_num,
    hidden_units,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = config_feat_data_small
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if task == "rating":
        with pytest.raises(AssertionError):
            _ = YouTubeRetrieval(task, data_info, loss_type)
    elif hidden_units in ("64,64", [1, 2, 4.22]):
        with pytest.raises(ValueError):
            _ = YouTubeRetrieval(task, data_info, hidden_units=hidden_units)
    elif neg_sampling == "wrong_labels":
        with pytest.raises(ValueError):
            YouTubeRetrieval(task, data_info, loss_type).fit(
                train_data, neg_sampling=False
            )
    elif task == "ranking" and loss_type not in ("nce", "sampled_softmax"):
        with pytest.raises(ValueError):
            YouTubeRetrieval(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = YouTubeRetrieval(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            norm_embed=norm_embed,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=10,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            num_sampled_per_batch=num_sampled_per_batch,
            recent_num=recent_num,
            random_num=random_num,
            tf_sess_config=None,
        )
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
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)
        seq_rec = ptest_seq_recommends(model, pd_data)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(
            YouTubeRetrieval, model, data_info
        )
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        loaded_seq_rec = ptest_seq_recommends(loaded_model, pd_data)
        assert_array_equal(seq_rec, loaded_seq_rec)

        remove_path(SAVE_PATH)


@pytest.mark.parametrize(
    "config_feat_data_small",
    [
        {
            "sparse_col": ["sex", "occupation"],
            "multi_sparse_col": [["genre1", "genre2", "genre3"]],
            "dense_col": ["age"],
            "user_col": ["sex", "occupation", "age", "genre1", "genre2", "genre3"],
        },
    ],
    indirect=True,
)
def test_youtube_retrieval_multi_sparse(config_feat_data_small):
    tf.compat.v1.reset_default_graph()
    task = "ranking"
    pd_data, train_data, eval_data, data_info = config_feat_data_small
    model = YouTubeRetrieval(
        task=task,
        data_info=data_info,
        loss_type="sampled_softmax",
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=True,
        reg=None,
        batch_size=10,
        use_bn=True,
        dropout_rate=None,
        num_sampled_per_batch=None,
        tf_sess_config=None,
    )
    model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=get_metrics(task),
        eval_user_num=200,
    )
    ptest_preds(model, task, pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=False)
    seq_rec = ptest_seq_recommends(model, pd_data)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(YouTubeRetrieval, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=False)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
    loaded_seq_rec = ptest_seq_recommends(loaded_model, pd_data)
    assert_array_equal(seq_rec, loaded_seq_rec)

    remove_path(SAVE_PATH)
