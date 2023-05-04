import numpy as np
import torch.linalg

from ..tfops import tf


def normalize_embeds(*embeds, backend):
    normed_embeds = []
    for e in embeds:
        if backend == "tf":
            ne = tf.linalg.l2_normalize(e, axis=1)
        elif backend == "torch":
            norms = torch.linalg.norm(e, dim=1, keepdim=True)
            ne = e / norms
        else:
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            ne = e / norms
        normed_embeds.append(ne)
    return normed_embeds[0] if len(embeds) == 1 else normed_embeds
