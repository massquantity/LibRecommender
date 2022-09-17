import tensorflow as tf

from libreco.bases import TfBase


def fit_multi_sparse(cls, train_data, eval_data, data_info, lr=None):
    if issubclass(cls, TfBase):
        tf.compat.v1.reset_default_graph()
    train_data.build_negative_samples(data_info, seed=2022)
    eval_data.build_negative_samples(data_info, seed=2222)
    model = cls(
        task="ranking",
        data_info=data_info,
        loss_type="cross_entropy",
        embed_size=4,
        n_epochs=1,
        lr=1e-4 if not lr else lr,
        batch_size=8192,
        eval_user_num=40,
    )
    model.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=["roc_auc", "precision", "map", "ndcg"],
    )
    return model
