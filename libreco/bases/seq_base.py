import os

import numpy as np

from .embed_base import EmbedBase
from ..batch.sequence import get_user_last_interacted
from ..embedding import normalize_embeds
from ..tfops import sess_config
from ..utils.save_load import load_tf_variables
from ..utils.validate import check_seq_mode


class SeqEmbedBase(EmbedBase):
    """Base class for sequence embedding models.

    These embedding models can make recommendation for arbitrary item sequence.
    So they also need to save the tf variables for inference.
    """

    def __init__(
        self,
        task,
        data_info,
        embed_size,
        norm_embed,
        recent_num,
        random_num,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)
        self.sess = sess_config(tf_sess_config)
        self.norm_embed = norm_embed
        self.seq_mode, self.max_seq_len = check_seq_mode(recent_num, random_num)
        self.recent_seqs, self.recent_seq_lens = self._set_recent_seqs()
        self.user_interacted_seq = None
        self.user_interacted_len = None
        self.user_embeds = None
        self.item_embeds = None
        self.item_biases = None

    def build_model(self):
        raise NotImplementedError

    def _set_recent_seqs(self):
        recent_seqs, recent_seq_lens = get_user_last_interacted(
            self.n_users, self.user_consumed, self.n_items, self.max_seq_len
        )
        return recent_seqs, recent_seq_lens.astype(np.int64)

    def set_embeddings(self):
        feed_dict = {
            self.user_interacted_seq: self.recent_seqs,
            self.user_interacted_len: self.recent_seq_lens,
        }
        if hasattr(self, "user_indices"):
            feed_dict[self.user_indices] = np.arange(self.n_users)
        user_embeds = self.sess.run(self.user_embeds, feed_dict)
        item_embeds = self.sess.run(self.item_embeds)
        item_biases = self.sess.run(self.item_biases)
        if self.norm_embed:
            user_embeds, item_embeds = normalize_embeds(
                user_embeds, item_embeds, backend="np"
            )

        user_biases = np.ones([len(user_embeds), 1], dtype=user_embeds.dtype)
        item_biases = item_biases[:, None]
        self.user_embeds_np = np.hstack([user_embeds, user_biases])
        self.item_embeds_np = np.hstack([item_embeds, item_biases])

    def save(self, path, model_name, inference_only=False, **_):
        super().save(path, model_name, inference_only=False)
        if inference_only:
            embed_path = os.path.join(path, model_name)
            np.savez_compressed(
                file=embed_path,
                user_embed=self.user_embeds_np,
                item_embed=self.item_embeds_np,
            )

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        model = load_tf_variables(cls, path, model_name, data_info)
        embeddings = np.load(os.path.join(path, f"{model_name}.npz"))
        model.user_embeds_np = embeddings["user_embed"]
        model.item_embeds_np = embeddings["item_embed"]
        return model
