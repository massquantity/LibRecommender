"""

Reference: Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
           (https://arxiv.org/pdf/1810.11921.pdf)

author: massquantity

"""
from tensorflow.keras.initializers import truncated_normal

from ..bases import TfBase
from ..tfops import (
    dropout_config,
    multi_sparse_combine_embedding,
    reg_config,
    tf,
    tf_dense,
)
from ..training import TensorFlowTrainer
from ..utils.validate import (
    check_dense_values,
    check_multi_sparse,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
    sparse_field_size,
    true_sparse_field_size,
)


class AutoInt(TfBase):
    user_variables = ["user_feat"]
    item_variables = ["item_feat"]
    sparse_variables = ["sparse_feat"]
    dense_variables = ["dense_feat"]

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="cross_entropy",
        embed_size=16,
        att_embed_size=None,
        num_heads=2,
        use_residual=True,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        hidden_units="128,64,32",
        multi_sparse_combiner="sqrtn",
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
        tf_sess_config=None,
        with_training=True,
    ):
        super().__init__(task, data_info, lower_upper_bound, tf_sess_config)

        self.all_args = locals()
        self.embed_size = embed_size
        # 'att_embed_size' also decides the num of attention layer
        self.att_embed_size, self.att_layer_num = self._att_config(att_embed_size)
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.reg = reg_config(reg)
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.seed = seed
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
        if self.sparse:
            self.sparse_feature_size = sparse_feat_size(data_info)
            self.sparse_field_size = sparse_field_size(data_info)
            self.multi_sparse_combiner = check_multi_sparse(
                data_info, multi_sparse_combiner
            )
            self.true_sparse_field_size = true_sparse_field_size(
                data_info, self.sparse_field_size, self.multi_sparse_combiner
            )
        if self.dense:
            self.dense_field_size = dense_field_size(data_info)
        self._build_model()
        if with_training:
            self.trainer = TensorFlowTrainer(
                self,
                task,
                loss_type,
                n_epochs,
                lr,
                lr_decay,
                batch_size,
                num_neg,
                k,
                eval_batch_size,
                eval_user_num,
            )

    def _build_model(self):
        tf.set_random_seed(self.seed)
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.concat_embed = []

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        attention_layer = tf.concat(self.concat_embed, axis=1)
        for i in range(self.att_layer_num):
            attention_layer = self.multi_head_attention(
                attention_layer, self.att_embed_size[i]
            )
        attention_layer = tf.layers.flatten(attention_layer)
        self.output = tf.squeeze(tf_dense(units=1)(attention_layer))

    def _build_user_item(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])

        user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users + 1, self.embed_size],
            initializer=truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        item_feat = tf.get_variable(
            name="item_feat",
            shape=[self.n_items + 1, self.embed_size],
            initializer=truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        user_embed = tf.expand_dims(
            tf.nn.embedding_lookup(user_feat, self.user_indices), axis=1
        )
        item_embed = tf.expand_dims(
            tf.nn.embedding_lookup(item_feat, self.item_indices), axis=1
        )
        self.concat_embed.extend([user_embed, item_embed])

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )

        sparse_feat = tf.get_variable(
            name="sparse_feat",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        if self.data_info.multi_sparse_combine_info and self.multi_sparse_combiner in (
            "sum",
            "mean",
            "sqrtn",
        ):
            sparse_embed = multi_sparse_combine_embedding(
                self.data_info,
                sparse_feat,
                self.sparse_indices,
                self.multi_sparse_combiner,
                self.embed_size,
            )
        else:
            sparse_embed = tf.nn.embedding_lookup(sparse_feat, self.sparse_indices)

        self.concat_embed.append(sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1]
        )

        dense_feat = tf.get_variable(
            name="dense_feat",
            shape=[self.dense_field_size, self.embed_size],
            initializer=truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        batch_size = tf.shape(self.dense_values)[0]
        # 1 * F_dense * K
        dense_embed = tf.expand_dims(dense_feat, axis=0)
        # B * F_dense * K
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        dense_embed = tf.multiply(dense_embed, dense_values_reshape)
        self.concat_embed.append(dense_embed)

    # inputs: B * F * Ki, new_embed_size: K, num_heads: H
    def multi_head_attention(self, inputs, new_embed_size):
        multi_embed_size = self.num_heads * new_embed_size
        # B * F * (K*H)
        queries = tf_dense(
            units=multi_embed_size,
            activation=None,
            kernel_initializer=truncated_normal(0.0, 0.01),
            use_bias=False,
        )(inputs)
        keys = tf_dense(
            units=multi_embed_size,
            activation=None,
            kernel_initializer=truncated_normal(0.0, 0.01),
            use_bias=False,
        )(inputs)
        values = tf_dense(
            units=multi_embed_size,
            activation=None,
            kernel_initializer=truncated_normal(0.0, 0.01),
            use_bias=False,
        )(inputs)
        if self.use_residual:
            residual = tf_dense(
                units=multi_embed_size,
                activation=None,
                kernel_initializer=truncated_normal(0.0, 0.01),
                use_bias=False,
            )(inputs)

        # H * B * F * K
        queries = tf.stack(tf.split(queries, self.num_heads, axis=2))
        keys = tf.stack(tf.split(keys, self.num_heads, axis=2))
        values = tf.stack(tf.split(values, self.num_heads, axis=2))

        # H * B * F * F
        weights = queries @ tf.transpose(keys, [0, 1, 3, 2])
        # weights = weights / np.sqrt(new_embed_size)
        weights = tf.nn.softmax(weights)
        # H * B * F * K
        outputs = weights @ values
        # 1 * B * F * (K*H)
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=-1)
        # B * F * (K*H)
        outputs = tf.squeeze(outputs, axis=0)
        if self.use_residual:
            # noinspection PyUnboundLocalVariable
            outputs += residual
        outputs = tf.nn.relu(outputs)
        return outputs

    @staticmethod
    def _att_config(att_embed_size):
        if not att_embed_size:
            att_embed_size = (8, 8, 8)
            att_layer_num = 3
        elif isinstance(att_embed_size, int):
            att_embed_size = [att_embed_size]
            att_layer_num = 1
        elif isinstance(att_embed_size, (list, tuple)):
            att_layer_num = len(att_embed_size)
        else:
            raise ValueError("att_embed_size must be int or list")
        return att_embed_size, att_layer_num
