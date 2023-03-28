import numpy as np

from ..utils.misc import colorize


def check_unknown(model, user, item):
    unknown_user_indices = list(np.where(user == model.n_users)[0])
    unknown_item_indices = list(np.where(item == model.n_items)[0])
    unknown_index = list(set(unknown_user_indices) | set(unknown_item_indices))
    unknown_num = len(unknown_index)
    if unknown_num > 0:
        unknown_str = (
            f"Detect {unknown_num} unknown interaction(s), position: {unknown_index}"
        )
        print(f"{colorize(unknown_str, 'red')}")
    return unknown_num, unknown_index, user, item


def check_unknown_user(data_info, user, inner_id=False):
    known_users_ids, unknown_users = [], []
    users = [user] if np.isscalar(user) else user
    for u in users:
        user_id = data_info.user2id.get(u, -1) if not inner_id else u
        if 0 <= user_id < data_info.n_users:
            known_users_ids.append(user_id)
        else:
            if not inner_id:
                unknown_str = f"Detect unknown user: {u}"
                print(f"{colorize(unknown_str, 'red')}")
            unknown_users.append(u)
    return known_users_ids, unknown_users


# def check_has_sampled(data, verbose):
#    if not data.has_sampled and verbose > 1:
#        exception_str = (
#            "During training, "
#            "one must do whole data sampling "
#            "before evaluating on epochs."
#        )
#        raise NotSamplingError(f"{colorize(exception_str, 'red')}")


def check_seq_mode(recent_num, random_num):
    if recent_num is not None:
        assert isinstance(recent_num, int), "recent_num must be integer"
        mode = "recent"
        num = recent_num
    elif random_num is not None:
        assert isinstance(random_num, int), "random_num must be integer"
        mode = "random"
        num = random_num
    else:
        mode = "recent"
        num = 10  # by default choose 10 recent interactions
    return mode, num


def check_sparse_indices(data_info):
    return False if not data_info.sparse_col.name else True


def check_dense_values(data_info):
    return False if not data_info.dense_col.name else True


def sparse_feat_size(data_info):
    if (
        data_info.user_sparse_unique is not None
        and data_info.item_sparse_unique is not None
    ):
        return (
            max(
                np.max(data_info.user_sparse_unique),
                np.max(data_info.item_sparse_unique),
            )
            + 1
        )
    elif data_info.user_sparse_unique is not None:
        return np.max(data_info.user_sparse_unique) + 1
    elif data_info.item_sparse_unique is not None:
        return np.max(data_info.item_sparse_unique) + 1


def sparse_field_size(data_info):
    return len(data_info.sparse_col.name)


def dense_field_size(data_info):
    return len(data_info.dense_col.name)


def check_multi_sparse(data_info, multi_sparse_combiner):
    if data_info.multi_sparse_combine_info and multi_sparse_combiner is not None:
        if multi_sparse_combiner not in ("normal", "sum", "mean", "sqrtn"):
            raise ValueError(
                f"unsupported multi_sparse_combiner type: {multi_sparse_combiner}"
            )
        else:
            combiner = multi_sparse_combiner
    else:
        combiner = "normal"
    return combiner


def check_fitting(model, train_data, eval_data, neg_sampling, k):
    check_neg_sampling(model, neg_sampling)
    check_labels(model, train_data.labels, neg_sampling)
    check_retrain_loaded_model(model)
    check_eval(eval_data, k, model.n_items)


def check_neg_sampling(model, neg_sampling):
    assert isinstance(neg_sampling, bool), (
        f"`neg_sampling` in `fit()` must be bool, got `{neg_sampling}`. "
        f"Set `model.fit(..., neg_sampling=True)` if your data is implicit(i.e., `task` is ranking) "
        f"and ONLY contains positive labels. Otherwise, negative sampling is not needed."
    )
    if model.task == "rating" and neg_sampling:
        raise ValueError("`rating` task should not use negative sampling")
    if (
        hasattr(model, "loss_type")
        and model.loss_type in ("bpr", "max_margin")
        and not neg_sampling
    ):
        raise ValueError(f"`{model.loss_type}` loss must use negative sampling.")


def check_labels(model, labels, neg_sampling):
    # still needs negative sampling when evaluating for these models
    # if model.model_name in (
    #    "YouTubeRetrieval",
    #    "UserCF",
    #    "ItemCF",
    #    "Item2Vec",
    #    "DeepWalk",
    # ):
    #    return
    if model.task == "ranking" and not neg_sampling:
        unique_labels = np.unique(labels)
        if (
            len(unique_labels) != 2
            or min(unique_labels) != 0.0
            or max(unique_labels) != 1.0
        ):
            raise ValueError(
                f"For `ranking` task without negative sampling, labels in data must be 0 and 1, "
                f"got unique labels: {unique_labels}"
            )


def check_retrain_loaded_model(model):
    if hasattr(model, "loaded") and model.loaded:
        raise RuntimeError(
            "Loaded model doesn't support retraining, use `rebuild_model` instead. "
            "Or constructing a new model from scratch."
        )


def check_eval(eval_data, k, n_items):
    if eval_data is not None and k > n_items:
        raise ValueError(f"eval `k` {k} exceeds num of items {n_items}")
