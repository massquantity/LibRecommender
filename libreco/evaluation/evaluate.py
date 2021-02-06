import numbers
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from ..data import TransformedSet
from .metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k
from .metrics import POINTWISE_METRICS, LISTWISE_METRICS, ALLOWED_METRICS
from .computation import (
    compute_preds,
    compute_probs,
    compute_recommends,
    build_transformed_data
)


class EvalMixin(object):
    def __init__(self, task, data_info, eval_class=None):
        self.task = task
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.eval_class = eval_class

    def _check_metrics(self, metrics, k):
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        if self.task == "rating":
            for m in metrics:
                if m not in ALLOWED_METRICS["rating_metrics"]:
                    raise ValueError(
                        f"metrics {m} is not suitable for rating task...")
        elif self.task == "ranking":
            for m in metrics:
                if m not in ALLOWED_METRICS["ranking_metrics"]:
                    raise ValueError(
                        f"metrics {m} is not suitable for ranking task...")

        if not isinstance(k, numbers.Integral):
            raise TypeError("k must be integer")

        return metrics

    def print_metrics(self, train_data=None, eval_data=None, metrics=None,
                      eval_batch_size=8192, k=10, sample_user_num=2048,
                      **kwargs):
        if not metrics:
            metrics = ["loss"]
        metrics = self._check_metrics(metrics, k)
        seed = kwargs.get("seed", 42)
        if "eval_batch_size" in kwargs:
            eval_batch_size = kwargs["eval_batch_size"]
        if "k" in kwargs:
            k = kwargs["k"]
        if "sample_user_num" in kwargs:
            sample_user_num = kwargs["sample_user_num"]

        if self.task == "rating":
            if train_data:
                y_pred, y_true = compute_preds(
                    self, train_data, eval_batch_size
                )
                # y_true = train_data.labels
                print_metrics_rating(
                    metrics, y_true, y_pred, train=True, **kwargs)
            if eval_data:
                y_pred, y_true = compute_preds(
                    self, eval_data, eval_batch_size
                )
                # y_true = eval_data.labels
                print_metrics_rating(
                    metrics, y_true, y_pred, train=False, **kwargs)

        elif self.task == "ranking":
            if train_data:
                train_params = dict()
                if POINTWISE_METRICS.intersection(metrics):
                    (train_params["y_prob"],
                     train_params["y_true"]) = compute_probs(
                        self, train_data, eval_batch_size
                    )
                    # train_params["y_true"] = train_data.labels

                print_metrics_ranking(metrics, **train_params, train=True)

            if eval_data:
                test_params = dict()
                if POINTWISE_METRICS.intersection(metrics):
                    (test_params["y_prob"],
                     test_params["y_true"]) = compute_probs(
                        self, eval_data, eval_batch_size
                    )
                    # test_params["y_true"] = eval_data.labels

                if LISTWISE_METRICS.intersection(metrics):
                    chosen_users = sample_user(
                        eval_data, seed, sample_user_num)
                    (test_params["y_reco_list"],
                     test_params["users"]) = compute_recommends(
                        self, chosen_users, k)
                    test_params["y_true_list"] = eval_data.user_consumed

                print_metrics_ranking(metrics, **test_params, k=k, train=False)


def sample_user(data, seed, num):
    np.random.seed(seed)
    unique_users = np.unique(data.user_indices)
    if isinstance(num, numbers.Integral) and num < len(unique_users):
        # noinspection PyTypeChecker
        users = np.random.choice(unique_users, num, replace=False)
    else:
        users = unique_users
    if isinstance(users, np.ndarray):
        users = list(users)
    return users


def evaluate(model, data, eval_batch_size=8192, metrics=None, k=10,
             sample_user_num=2048, neg_sample=False, update_features=False,
             **kwargs):
    seed = kwargs.get("seed", 42)
    if isinstance(data, pd.DataFrame):
        data = build_transformed_data(
            model, data, neg_sample, update_features, seed
        )
    assert isinstance(data, TransformedSet), (
        "The data from evaluation must be TransformedSet object."
    )
    if not metrics:
        metrics = ["loss"]
    metrics = model._check_metrics(metrics, k)
    eval_result = dict()

    if model.task == "rating":
        y_pred, y_true = compute_preds(model, data, eval_batch_size,)
        for m in metrics:
            if m in ["rmse", "loss"]:
                eval_result[m] = np.sqrt(
                    mean_squared_error(y_true, y_pred))
            elif m == "mae":
                eval_result[m] = mean_absolute_error(y_true, y_pred)
            elif m == "r2":
                eval_result[m] = r2_score(y_true, y_pred)

    elif model.task == "ranking":
        if POINTWISE_METRICS.intersection(metrics):
            y_prob, y_true = compute_probs(model, data, eval_batch_size,)
        if LISTWISE_METRICS.intersection(metrics):
            chosen_users = sample_user(data, seed, sample_user_num)
            y_reco_list, users = compute_recommends(model, chosen_users, k)
            y_true_list = data.user_consumed

        for m in metrics:
            if m in ["log_loss", "loss"]:
                eval_result[m] = log_loss(y_true, y_prob, eps=1e-7)
            elif m == "balanced_accuracy":
                y_pred = np.round(y_prob)
                eval_result[m] = balanced_accuracy_score(y_true, y_pred)
            elif m == "roc_auc":
                eval_result[m] = roc_auc_score(y_true, y_prob)
            elif m == "pr_auc":
                precision, recall, _ = precision_recall_curve(y_true,
                                                              y_prob)
                eval_result[m] = auc(recall, precision)
            elif m == "precision":
                eval_result[m] = precision_at_k(y_true_list,
                                                y_reco_list,
                                                users, k)
            elif m == "recall":
                eval_result[m] = recall_at_k(y_true_list,
                                             y_reco_list,
                                             users, k)
            elif m == "map":
                eval_result[m] = map_at_k(y_true_list,
                                          y_reco_list,
                                          users, k)
            elif m == "ndcg":
                eval_result[m] = ndcg_at_k(y_true_list,
                                           y_reco_list,
                                           users, k)

    return eval_result


def print_metrics_rating(metrics, y_true, y_pred, train=True, **kwargs):
    if kwargs.get("lower_bound") and kwargs.get("upper_bound"):
        lower_bound, upper_bound = (
            kwargs.get("lower_bound"), kwargs.get("upper_bound"))
        y_pred = np.clip(y_pred, lower_bound, upper_bound)
    if train:
        for m in metrics:
            if m in ["rmse", "loss"]:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                print(f"\t train rmse: {rmse:.4f}")
    else:
        for m in metrics:
            if m in ["rmse", "loss"]:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                print(f"\t eval rmse: {rmse:.4f}")
            elif m == "mae":
                mae = mean_absolute_error(y_true, y_pred)
                print(f"\t eval mae: {mae:.4f}")
            elif m == "r2":
                r_squared = r2_score(y_true, y_pred)
                print(f"\t eval r2: {r_squared:.4f}")


def print_metrics_ranking(metrics, y_prob=None, y_true=None, y_reco_list=None,
                          y_true_list=None, users=None, k=10, train=True):
    if train:
        for m in metrics:
            if m in ["log_loss", "loss"]:
                log_loss_ = log_loss(y_true, y_prob, eps=1e-7)
                print(f"\t train log_loss: {log_loss_:.4f}")
    else:
        for m in metrics:
            if m in ["log_loss", "loss"]:
                log_loss_ = log_loss(y_true, y_prob, eps=1e-7)
                print(f"\t eval log_loss: {log_loss_:.4f}")
            elif m == "balanced_accuracy":
                y_pred = np.round(y_prob)
                accuracy = balanced_accuracy_score(y_true, y_pred)
                print(f"\t eval balanced accuracy: {accuracy:.4f}")
            elif m == "roc_auc":
                roc_auc = roc_auc_score(y_true, y_prob)
                print(f"\t eval roc_auc: {roc_auc:.4f}")
            elif m == "pr_auc":
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = auc(recall, precision)
                print(f"\t eval pr_auc: {pr_auc:.4f}")
            elif m == "precision":
                precision_all = precision_at_k(y_true_list, y_reco_list,
                                               users, k)
                print(f"\t eval precision@{k}: {precision_all:.4f}")
            elif m == "recall":
                recall_all = recall_at_k(y_true_list, y_reco_list, users, k)
                print(f"\t eval recall@{k}: {recall_all:.4f}")
            elif m == "map":
                map_all = map_at_k(y_true_list, y_reco_list, users, k)
                print(f"\t eval map@{k}: {map_all:.4f}")
            elif m == "ndcg":
                ndcg_all = ndcg_at_k(y_true_list, y_reco_list, users, k)
                print(f"\t eval ndcg@{k}: {ndcg_all:.4f}")

