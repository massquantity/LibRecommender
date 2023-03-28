import tensorflow as tf

from libreco.bases import TfBase


def fit_multi_sparse(cls, train_data, eval_data, data_info, lr=None):
    if issubclass(cls, TfBase):
        tf.compat.v1.reset_default_graph()

    model = cls(
        task="ranking",
        data_info=data_info,
        loss_type="cross_entropy",
        embed_size=4,
        n_epochs=1,
        lr=1e-4 if not lr else lr,
        batch_size=100,
    )
    model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=["roc_auc", "precision", "map", "ndcg"],
        eval_user_num=40,
    )
    return model
