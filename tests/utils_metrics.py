def get_metrics(task):
    if task == "rating":
        return ["rmse", "mae", "r2"]
    else:
        return [
            "loss",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "map",
            "ndcg",
        ]
