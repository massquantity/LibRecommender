"""Implementation of Caser."""
from ..bases import ModelMeta, SeqEmbedBase
from ..tfops import conv_nn, dropout_config, max_pool, reg_config, tf, tf_dense
from ..utils.misc import count_params


class Caser(SeqEmbedBase, metaclass=ModelMeta, backend="tensorflow"):
    """*Caser* algorithm.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal'}, default: 'cross_entropy'
        Loss for model training.
    embed_size: int, default: 16
        Vector size of embeddings.
    n_epochs: int, default: 10
        Number of epochs for training.
    lr : float, default 0.001
        Learning rate for training.
    lr_decay : bool, default: False
        Whether to use learning rate decay.
    epsilon : float, default: 1e-5
        A small constant added to the denominator to improve numerical stability in
        Adam optimizer.
        According to the `official comment <https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64>`_,
        default value of `1e-8` for `epsilon` is generally not good, so here we choose `1e-5`.
        Users can try tuning this hyperparameter if the training is unstable.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    batch_size : int, default: 256
        Batch size for training.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative sampling strategy.

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.

        .. versionadded:: 1.1.0

    num_neg : int, default: 1
        Number of negative samples for each positive sample, only used in `ranking` task.
    use_bn : bool, default: True
        Whether to use batch normalization.
    dropout_rate : float or None, default: None
        Probability of an element to be zeroed. If it is None, dropout is not used.
    nh_filters : int, default: 2
        Number of output filters in the horizontal CNN layer.
    nv_filters : int, default: 4
        Number of output filters in the vertical CNN layer.
    recent_num : int or None, default: 10
        Number of recent items to use in user behavior sequence.
    random_num : int or None, default: None
        Number of random sampled items to use in user behavior sequence.
        If `recent_num` is not None, `random_num` is not considered.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    References
    ----------
    *Jiaxi Tang & Ke Wang.* `Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
    <https://arxiv.org/pdf/1809.07426.pdf>`_.
    """

    user_variables = ["user_feat"]
    item_variables = ["item_weights", "item_biases", "input_embed"]

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
        use_bn=False,
        dropout_rate=None,
        nh_filters=2,
        nv_filters=4,
        recent_num=10,
        random_num=None,
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(
            task,
            data_info,
            embed_size,
            recent_num,
            random_num,
            lower_upper_bound,
            tf_sess_config,
        )
        self.all_args = locals()
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.dropout_rate = dropout_config(dropout_rate)
        self.use_bn = use_bn
        self.nh_filters = nh_filters
        self.nv_filters = nv_filters
        self.seed = seed

    def build_model(self):
        tf.set_random_seed(self.seed)
        self._build_placeholders()
        self._build_variables()
        self._build_user_embeddings()

        item_vector = tf.nn.embedding_lookup(self.item_weights, self.item_indices)
        item_bias = tf.nn.embedding_lookup(self.item_biases, self.item_indices)
        self.output = (
            tf.reduce_sum(tf.multiply(self.user_vector, item_vector), axis=1)
            + item_bias
        )
        count_params()

    def _build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.max_seq_len]
        )
        self.user_interacted_len = tf.placeholder(tf.int64, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

    def _build_variables(self):
        self.user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        # weight and bias parameters for last fc_layer
        self.item_biases = tf.get_variable(
            name="item_biases",
            shape=[self.n_items],
            initializer=tf.zeros_initializer(),
        )
        self.item_weights = tf.get_variable(
            name="item_weights",
            shape=[self.n_items, self.embed_size * 2],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        # input_embed for cnn_layer, include padding value
        self.input_embed = tf.get_variable(
            name="input_embed",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

    def _build_user_embeddings(self):
        user_repr = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        # B * seq * K
        seq_item_embed = tf.nn.embedding_lookup(
            self.input_embed, self.user_interacted_seq
        )

        convs_out = []
        for i in range(1, self.max_seq_len + 1):
            # h_conv = tf.layers.conv1d(
            #    inputs=seq_item_embed,
            #    filters=self.nh_filters,
            #    kernel_size=i,
            #    strides=1,
            #    padding="valid",
            #    activation=tf.nn.relu
            # )

            h_conv = conv_nn(
                filters=self.nh_filters,
                kernel_size=i,
                strides=1,
                padding="valid",
                activation="relu",
            )(inputs=seq_item_embed)

            # h_conv = tf.reduce_max(h_conv, axis=1)
            h_size = h_conv.get_shape().as_list()[1]
            h_conv = max_pool(pool_size=h_size, strides=1, padding="valid")(h_conv)
            h_conv = tf.squeeze(h_conv, axis=1)
            convs_out.append(h_conv)

        v_conv = conv_nn(
            filters=self.nv_filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            activation="relu",
        )(inputs=tf.transpose(seq_item_embed, [0, 2, 1]))
        convs_out.append(tf.keras.layers.Flatten()(v_conv))

        convs_out = tf.concat(convs_out, axis=1)
        convs_out = tf_dense(units=self.embed_size, activation=tf.nn.relu)(convs_out)
        self.user_vector = tf.concat([user_repr, convs_out], axis=1)
