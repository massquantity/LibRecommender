import os

import numpy as np

from .embed_base import EmbedBase
from ..batch.sequence import get_recent_seqs
from ..layers import normalize_embeds
from ..recommendation import check_dynamic_rec_feats, rank_recommendations
from ..recommendation.preprocess import process_embed_feat, process_embed_seq
from ..tfops import get_variable_from_graph, sess_config, tf
from ..tfops.features import get_feed_dict
from ..utils.constants import SequenceModels
from ..utils.save_load import load_tf_variables
from ..utils.validate import check_seq_mode


class DynEmbedBase(EmbedBase):
    """Base class for dynamic embedding models.

    These models can generate embedding and make recommendation based on
    arbitrary user features or item sequences.
    So they also need to save the tf variables for inference.

    .. versionadded:: 1.2.0

    """

    def __init__(
        self,
        task,
        data_info,
        embed_size,
        norm_embed,
        recent_num=None,
        random_num=None,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)
        self.sess = sess_config(tf_sess_config)
        self.norm_embed = norm_embed
        self.user_embeds = None
        self.item_embeds = None
        self.item_biases = None
        if (
            SequenceModels.contains(self.model_name)
            and self.model_name != "YouTubeRetrieval"
        ):
            self.seq_mode, self.max_seq_len = check_seq_mode(recent_num, random_num)
            self.recent_seqs, self.recent_seq_lens = get_recent_seqs(
                self.n_users,
                self.user_consumed,
                self.n_items,
                self.max_seq_len,
            )

    def build_model(self):
        raise NotImplementedError

    def convert_array_id(self, user, inner_id):
        """Convert a single user to inner user id.

        If the user doesn't exist, it will be converted to padding id.
        The return type should be `array_like` for further shape compatibility.
        """
        assert np.isscalar(user), f"User to convert must be scalar, got: {user}"
        if inner_id:
            if not isinstance(user, (int, np.integer)):
                raise ValueError(f"`inner id` user must be int, got {user}")
            return np.array([user if 0 <= user < self.n_users else self.n_users])
        else:
            return np.array([self.data_info.user2id.get(user, self.n_users)])

    def recommend_user(
        self,
        user,
        n_rec,
        user_feats=None,
        seq=None,
        cold_start="average",
        inner_id=False,
        filter_consumed=True,
        random_rec=False,
    ):
        """Recommend a list of items for given user(s).

        If both ``user_feats`` and ``seq`` are ``None``, the model will use the precomputed
        embeddings for recommendation, and the ``cold_start`` strategy will be used for unknown users.

        If either ``user_feats`` or ``seq`` is provided, the model will generate user embedding
        dynamically for recommendation. In this case, if the ``user`` is unknown,
        it will be set to padding id, which means the ``cold_start`` strategy will not be applied.
        This situation is common when one wants to recommend for an unknown user based on
        user features or behavior sequence.

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids to recommend.
        n_rec : int
            Number of recommendations to return.
        user_feats : dict or None, default: None
            Extra user features for recommendation.

            .. versionadded:: 1.2.0

        seq : list or numpy.ndarray or None, default: None
            Extra item sequence for recommendation. If the sequence length is larger than
            `recent_num` hyperparameter specified in the model, it will be truncated.
            If smaller, it will be padded.

            .. versionadded:: 1.1.0

        cold_start : {'popular', 'average'}, default: 'average'
            Cold start strategy.

            - 'popular' will sample from popular items.
            - 'average' will use the average of all the user/item embeddings as the
              representation of the cold-start user/item.

        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`. For library users inner_id
            may never be used.
        filter_consumed : bool, default: True
            Whether to filter out items that a user has previously consumed.
        random_rec : bool, default: False
            Whether to choose items for recommendation based on their prediction scores.

        Returns
        -------
        recommendation : dict of {Union[int, str, array_like] : numpy.ndarray}
            Recommendation result with user ids as keys and array_like recommended items as values.
        """
        if user_feats is None and seq is None:
            return super().recommend_user(
                user, n_rec, cold_start, inner_id, filter_consumed, random_rec
            )

        check_dynamic_rec_feats(self.model_name, user, user_feats, seq)
        user_embed = self.dyn_user_embedding(
            user, user_feats=user_feats, seq=seq, include_bias=True, inner_id=inner_id
        )
        if user_embed.ndim == 1:
            user_embed = np.expand_dims(user_embed, axis=0)
        item_embeds = self.item_embeds_np[: self.n_items]
        preds = user_embed @ item_embeds.T

        computed_recs = rank_recommendations(
            self.task,
            self.convert_array_id(user, inner_id),
            preds,
            n_rec,
            self.n_items,
            self.user_consumed,
            filter_consumed,
            random_rec,
        )
        rec_items = (
            computed_recs[0]
            if inner_id
            else np.array([self.data_info.id2item[i] for i in computed_recs[0]])
        )
        # only one user is allowed in dynamic situation
        return {user: rec_items}

    def dyn_user_embedding(
        self,
        user,
        user_feats=None,
        seq=None,
        include_bias=False,
        inner_id=False,
    ):
        """Generate user embedding based on given user features or item sequence.

        .. versionadded:: 1.2.0

        Parameters
        ----------
        user : int or str
            Query user id. Must be a single user.
        user_feats : dict or None, default: None
            Extra user features for recommendation.
        seq : list or numpy.ndarray or None, default: None
            Extra item sequence for recommendation. If the sequence length is larger than
            `recent_num` hyperparameter specified in the model, it will be truncated.
            If smaller, it will be padded.
        include_bias : bool, default: False
            Whether to include bias term in returned embeddings.
            Note some models such as `SVD`, `BPR` etc., use bias term in model inference.
        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`. For library users inner_id
            may never be used.

        Returns
        -------
        user_embedding : numpy.ndarray
            Generated dynamic user embeddings.

        Raises
        ------
        ValueError
            If `user` is not a single user.
        ValueError
            If `seq` is provided but the model doesn't support sequence recommendation.
        """
        check_dynamic_rec_feats(self.model_name, user, user_feats, seq)
        if user is None:
            user_id, user_indices = None, np.arange(self.n_users)
        else:
            user_id = user_indices = self.convert_array_id(user, inner_id)

        sparse_indices, dense_values = process_embed_feat(
            self.data_info, user_id, user_feats
        )
        if SequenceModels.contains(self.model_name):
            seq, seq_len = process_embed_seq(self, user_id, seq, inner_id)
        else:
            seq = seq_len = None

        feed_dict = get_feed_dict(
            model=self,
            user_indices=user_indices,
            user_sparse_indices=sparse_indices,
            user_dense_values=dense_values,
            user_interacted_seq=seq,
            user_interacted_len=seq_len,
            is_training=False,
        )
        user_embeds = self.sess.run(self.user_embeds, feed_dict)
        # already normalized if specified in `TwoTower`
        if self.norm_embed and self.model_name != "TwoTower":
            user_embeds = normalize_embeds(user_embeds, backend="np")
        if include_bias and self.item_biases is not None:
            # add pseudo bias
            user_biases = np.ones([len(user_embeds), 1], dtype=user_embeds.dtype)
            user_embeds = np.hstack([user_embeds, user_biases])
        return user_embeds if user_id is None else np.squeeze(user_embeds, axis=0)

    def set_embeddings(self):
        self._assign_user_oov(var_name="user_embeds_var", scope_name="embedding")
        self.user_embeds_np = self.dyn_user_embedding(user=None, include_bias=True)

        if self.model_name != "TwoTower":
            feed_dict = None
        else:
            item_indices = np.arange(self.n_items)
            sparse_indices = dense_values = None
            if self.data_info.item_sparse_unique is not None:
                sparse_indices = self.data_info.item_sparse_unique[:-1]
            if self.data_info.item_dense_unique is not None:
                dense_values = self.data_info.item_dense_unique[:-1]

            feed_dict = get_feed_dict(
                self,
                item_indices=item_indices,
                item_sparse_indices=sparse_indices,
                item_dense_values=dense_values,
                is_training=False,
            )

        item_embeds = self.sess.run(self.item_embeds, feed_dict)
        # already normalized if specified in `TwoTower`
        if self.norm_embed and self.model_name != "TwoTower":
            item_embeds = normalize_embeds(item_embeds, backend="np")
        if self.item_biases is not None:
            item_biases = self.sess.run(self.item_biases)[:, None]
            item_embeds = np.hstack([item_embeds, item_biases])
        self.item_embeds_np = item_embeds

    def _assign_user_oov(self, var_name, scope_name):
        """Assign mean user embedding to padding index, used in cold-start scenario."""
        try:
            user_embeds_var = get_variable_from_graph(var_name, scope_name)
            mean_op = tf.IndexedSlices(
                tf.reduce_mean(
                    tf.gather(user_embeds_var, tf.range(self.n_users)),
                    axis=0,
                    keepdims=True,
                ),
                [self.n_users],
            )
            self.sess.run(user_embeds_var.scatter_update(mean_op))
        except ValueError:
            if hasattr(self, "user_variables"):
                print(
                    f"Failed to assign oov in user embeds, `{var_name}` doesn't exist."
                )
                raise

    def build_topk(self):
        self.k = tf.placeholder(tf.int32, shape=(), name="k")
        if self.norm_embed and self.model_name != "TwoTower":
            user_embeds, item_embeds = normalize_embeds(
                self.user_embeds, self.item_embeds, backend="tf"
            )
        else:
            user_embeds, item_embeds = self.user_embeds, self.item_embeds
        user_embeds = tf.squeeze(user_embeds, axis=0)
        preds = tf.linalg.matvec(item_embeds, user_embeds)
        if self.item_biases is not None:
            preds += self.item_biases
        _, indices = tf.math.top_k(preds, self.k, sorted=True)
        return indices

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
