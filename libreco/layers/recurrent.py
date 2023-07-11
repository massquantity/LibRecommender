from ..tfops import get_tf_version, tf


def tf_rnn(
    inputs,
    rnn_type,
    lengths,
    maxlen,
    hidden_units,
    dropout_rate,
    use_ln,
    is_training,
    version=None,
):
    tf_version = get_tf_version(version)
    if tf_version >= "2.0.0":
        # cell_type = (
        #    tf.keras.layers.LSTMCell
        #    if self.rnn_type.endswith("lstm")
        #    else tf.keras.layers.GRUCell
        # )
        # cells = [cell_type(size) for size in self.hidden_units]
        # masks = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
        # tf2_rnn = tf.keras.layers.RNN(cells, return_state=True)
        # output, *state = tf2_rnn(seq_item_embed, mask=masks)

        rnn_layer = (
            tf.keras.layers.LSTM if rnn_type.endswith("lstm") else tf.keras.layers.GRU
        )
        output = inputs
        masks = tf.sequence_mask(lengths, maxlen)
        for units in hidden_units:
            output = rnn_layer(
                units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                activation=None if use_ln else "tanh",
            )(output, mask=masks, training=is_training)

            if use_ln:
                output = tf.keras.layers.LayerNormalization()(output)
                output = tf.keras.activations.get("tanh")(output)

        return output[:, -1, :]

    else:
        cell_type = (
            tf.nn.rnn_cell.LSTMCell
            if rnn_type.endswith("lstm")
            else tf.nn.rnn_cell.GRUCell
        )
        cells = [cell_type(size) for size in hidden_units]
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        zero_state = stacked_cells.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
        _, state = tf.nn.dynamic_rnn(
            cell=stacked_cells,
            inputs=inputs,
            sequence_length=lengths,
            initial_state=zero_state,
            time_major=False,
        )
        return state[-1][1] if rnn_type == "lstm" else state[-1]
