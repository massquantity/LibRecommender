import os

import numpy as np

from .embed_base import EmbedBase
from ..batch.sequence import get_user_last_interacted
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
        recent_num,
        random_num,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)
        self.sess = sess_config(tf_sess_config)
        self.seq_mode, self.max_seq_len = check_seq_mode(recent_num, random_num)
        self.recent_seqs, self.recent_seq_lens = self._set_recent_seqs()
        self.user_interacted_seq = None
        self.user_interacted_len = None
        self.user_vector = None
        self.item_weights = None
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
        user_vector = self.sess.run(self.user_vector, feed_dict)
        item_weights = self.sess.run(self.item_weights)
        item_biases = self.sess.run(self.item_biases)

        user_bias = np.ones([len(user_vector), 1], dtype=user_vector.dtype)
        item_bias = item_biases[:, None]
        self.user_embed = np.hstack([user_vector, user_bias])
        self.item_embed = np.hstack([item_weights, item_bias])

    def save(self, path, model_name, inference_only=False, **_):
        super().save(path, model_name, inference_only=False)
        if inference_only:
            embed_path = os.path.join(path, model_name)
            np.savez_compressed(
                file=embed_path,
                user_embed=self.user_embed,
                item_embed=self.item_embed,
            )

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        model = load_tf_variables(cls, path, model_name, data_info)
        embeddings = np.load(os.path.join(path, f"{model_name}.npz"))
        model.user_embed = embeddings["user_embed"]
        model.item_embed = embeddings["item_embed"]
        return model
