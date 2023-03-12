import torch
import torch.nn.functional as F


def binary_cross_entropy_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels)


# focal loss for binary cross entropy based on [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf)
def focal_loss(logits, labels, alpha=0.25, gamma=2.0, mean=True):
    weighting_factor = (labels * alpha) + ((1 - labels) * (1 - alpha))
    probs = torch.sigmoid(logits)
    p_t = (labels * probs) + ((1 - labels) * (1 - probs))
    modulating_factor = torch.pow(1.0 - p_t, gamma)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    focal = weighting_factor * modulating_factor * bce
    if mean:
        focal = torch.mean(focal)
    return focal


def bpr_loss(pos_scores, neg_scores):
    log_sigmoid = F.logsigmoid(pos_scores - neg_scores)
    return torch.negative(torch.mean(log_sigmoid))


def max_margin_loss(pos_scores, neg_scores, margin):
    return F.margin_ranking_loss(
        pos_scores, neg_scores, torch.ones_like(pos_scores), margin=margin
    )


def pairwise_bce_loss(pos_scores, neg_scores, mean=True):
    pos_bce = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores), reduction="none"
    )
    neg_bce = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores), reduction="none"
    )
    pos_bce = _check_dim(pos_bce)
    neg_bce = _check_dim(neg_bce)
    loss = torch.cat([pos_bce, neg_bce])
    if mean:
        return torch.mean(loss)
    else:
        return torch.sum(loss)


def pairwise_focal_loss(pos_scores, neg_scores, mean=True):
    pos_focal = focal_loss(pos_scores, torch.ones_like(pos_scores), mean=False)
    neg_focal = focal_loss(neg_scores, torch.zeros_like(neg_scores), mean=False)
    pos_focal = _check_dim(pos_focal)
    neg_focal = _check_dim(neg_focal)
    loss = torch.cat([pos_focal, neg_focal])
    if mean:
        return torch.mean(loss)
    else:
        return torch.sum(loss)


def _check_dim(tensor):  # pragma: no cover
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def compute_pair_scores(targets, items_pos, items_neg, repeat_positives=True):
    if len(targets) == len(items_pos) == len(items_neg):
        pos_scores = torch.sum(torch.mul(targets, items_pos), dim=1)
        neg_scores = torch.sum(torch.mul(targets, items_neg), dim=1)
        return pos_scores, neg_scores

    if len(targets) != len(items_pos):
        raise ValueError(
            f"targets and items_pos length doesn't match, "
            f"got {len(targets)} and {len(items_pos)}"
        )
    pos_len, neg_len = len(items_pos), len(items_neg)
    if neg_len % pos_len != 0:
        raise ValueError(
            f"negatives length is not a multiple of positives length, "
            f"got {neg_len} and {pos_len}"
        )
    factor = int(neg_len / pos_len)
    embed_size = items_pos.shape[1]
    targets_neg = targets.unsqueeze(1)
    items_neg = items_neg.view(pos_len, factor, embed_size)
    pos_scores = torch.sum(torch.mul(targets, items_pos), dim=1)
    if repeat_positives:
        pos_scores = pos_scores.repeat_interleave(factor)
    neg_scores = torch.mul(targets_neg, items_neg).view(-1, embed_size)
    neg_scores = torch.sum(neg_scores, dim=1)
    return pos_scores, neg_scores
