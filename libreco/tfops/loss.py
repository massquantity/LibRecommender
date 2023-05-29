from .version import tf


def choose_tf_loss(model, task, loss_type):
    if task == "rating":
        loss = tf.losses.mean_squared_error(
            labels=model.labels, predictions=model.output
        )
    else:
        if loss_type == "cross_entropy":
            assert hasattr(model, "output"), (
                f"Binary cross entropy loss is unavailable in `{model.model_name}`"
            )  # fmt: skip
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=model.labels, logits=model.output
                )
            )
        elif loss_type == "bpr":
            assert hasattr(model, "bpr_loss"), (
                f"Bpr loss is unavailable in {model.model_name}"
            )  # fmt: skip
            loss = -tf.reduce_mean(model.bpr_loss)
        elif loss_type == "focal":
            loss = tf.reduce_mean(focal_loss(labels=model.labels, logits=model.output))
        elif loss_type == "max_margin":
            loss = tf.reduce_mean(
                max_margin_loss(
                    model.user_embeds,
                    model.item_embeds,
                    model.item_embeds_neg,
                    model.margin,
                )
            )
        elif loss_type == "softmax":
            loss = tf.reduce_mean(
                softmax_cross_entropy(model, model.user_embeds, model.item_embeds)
            )
            if hasattr(model, "ssl_pattern") and model.ssl_pattern is not None:
                ssl_loss = tf.reduce_mean(
                    softmax_cross_entropy(
                        model,
                        model.ssl_left_embeds,
                        model.ssl_right_embeds,
                        all_adjust=False,
                    )
                )
                loss += model.alpha * ssl_loss
        else:
            raise ValueError(f"unknown loss_type for ranking: {loss_type}")

    return loss


# focal loss for binary cross entropy based on [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf)
def focal_loss(labels, logits, alpha=0.25, gamma=2.0):
    weighting_factor = (labels * alpha) + ((1 - labels) * (1 - alpha))
    probs = tf.sigmoid(logits)
    p_t = (labels * probs) + ((1 - labels) * (1 - probs))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return weighting_factor * modulating_factor * bce


def max_margin_loss(user_embeds, item_embeds, item_embeds_neg, margin):
    pos_scores = tf.reduce_sum(user_embeds * item_embeds, axis=1)
    neg_scores = tf.reduce_sum(user_embeds * item_embeds_neg, axis=1)
    return tf.nn.relu(margin + neg_scores - pos_scores)


def softmax_cross_entropy(model, user_embeds, item_embeds, all_adjust=True):
    logits = tf.matmul(user_embeds, item_embeds, transpose_b=True)
    logits = model.adjust_logits(logits, all_adjust)
    labels = tf.range(tf.shape(user_embeds)[0])
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
