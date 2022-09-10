from .version import tf


def choose_tf_loss(model, task, loss_type):
    if task == "rating":
        loss = tf.losses.mean_squared_error(
            labels=model.labels, predictions=model.output
        )
    elif task == "ranking":
        if loss_type == "cross_entropy":
            assert hasattr(
                model, "output"
            ), f"can't use cross entropy loss in {model.model_name}"
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=model.labels, logits=model.output
                )
            )
        elif loss_type == "bpr":
            assert hasattr(
                model, "bpr_loss"
            ), f"can't use bpr loss in {model.model_name}"
            loss = -tf.reduce_mean(model.bpr_loss)
        elif loss_type == "focal":
            loss = tf.reduce_mean(focal_loss(labels=model.labels, logits=model.output))
        else:
            raise ValueError(f"unknown loss_type for ranking: {loss_type}")
    else:
        raise ValueError("task must be `rating` or `ranking`.")
    return loss


# focal loss for binary cross entropy based on [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf)
def focal_loss(labels, logits, alpha=0.25, gamma=2.0):
    weighting_factor = (labels * alpha) + ((1 - labels) * (1 - alpha))
    probs = tf.sigmoid(logits)
    p_t = (labels * probs) + ((1 - labels) * (1 - probs))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return weighting_factor * modulating_factor * bce
