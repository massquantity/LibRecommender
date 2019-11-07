import tensorflow as tf


def export_TFRecord():
    with tf.io.TFRecordWriter("tfrecord_1200w/train_all_ranking.tfrecord") as w:
        for i in range(0, len(user_real_indices)):
            user = int(user_real_indices[i])
            v_implicit = get_implicit_feedback_single(user)
            tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "item_indices": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[int(item_real_indices[i])])),
                        "values_implicit": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=v_implicit)),
                        "label": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[1.0])),
                        "feature_indices": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=feature_real_indices[i])),
                    }))
            w.write(tf_example.SerializeToString())

            item_neg = np.random.randint(0, n_items)
            while item_neg in u_items_dict[user]:
                item_neg = np.random.randint(0, n_items)

            fi = feature_real_indices[i].tolist()
            dt = item_indices_dict[item_neg]
            for col, orig_col in enumerate([3, 4, 5]):
                fi[orig_col] = dt[col]

            tf_example_neg = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "item_indices": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[item_neg])),
                        "values_implicit": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=v_implicit)),
                        "label": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[0.0])),
                        "feature_indices": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=fi)),
                    }))
            w.write(tf_example_neg.SerializeToString())