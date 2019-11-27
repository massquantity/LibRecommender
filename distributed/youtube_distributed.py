import os
import time
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import feature_column as fc


flags = tf.app.flags
flags.DEFINE_string("saved_model_dir", "./saved_model_dir", "Path to the saved model.")
flags.DEFINE_string("output_item_vector", "./output_item_vector/nce_weights.txt", "Path to the trained item vector.")
flags.DEFINE_string("dataset", "", "Directory for the dataset.")
flags.DEFINE_integer("n_classes", 3000, "num of possible classes/labels.")
flags.DEFINE_integer("num_sampled", 500, "num of negative samples")
flags.DEFINE_string("hidden_units", "128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("last_hidden_units", 64, "last hidden layer of the NN, equal to user vector")
flags.DEFINE_string("eval_top_n", "5,20,50", "Comma-separated list of top evaluate numbers")
flags.DEFINE_integer("train_steps", 200000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("top_k", 20, "predict the top k results")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
# flags.DEFINE_float("dropout_rate", 0.0, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
# flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_boolean("use_bn", False, "Whether to use batch normalization for hidden layers")
flags.DEFINE_boolean("predict", False, "Whether to predict")
flags.DEFINE_string("ps_hosts", "localhost:2221,localhost:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job_name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task that performs the variable "
                     "initialization ")
flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")  # norun_on_cluster
FLAGS = flags.FLAGS


def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name: %s" % FLAGS.job_name)
    print("task index: %d" % FLAGS.task_index)
    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}
    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)


def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        if task_type == "ps":
            tf_config["task"] = {"index": task_index, "type": task_type}
            FLAGS.job_name = "ps"
            FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG: ", json.loads(os.environ["TF_CONFIG"]))


def create_feature_columns(train_data):
    n_users = train_data.user.nunique()
    users = fc.categorical_column_with_vocabulary_list("user", np.arange(n_users), default_value=-1, dtype=tf.int64)
    gender = fc.categorical_column_with_vocabulary_list("gender", ["M", "F"])
    age = fc.categorical_column_with_vocabulary_list("age", [1, 18, 25, 35, 45, 50, 56], dtype=tf.int64)
    occupation = fc.categorical_column_with_vocabulary_list("occupation", np.arange(21), dtype=tf.int64)

    all_feature_cols = [fc.embedding_column(users, 32),
                        fc.indicator_column(gender),
                        fc.embedding_column(age, 32),
                        fc.embedding_column(occupation, 32)]

    return all_feature_cols


def input_fn(data, repeat=1, batch_size=256, mode="train"):
    if mode == "train":
        features = {"user": data["user"].values,
                    "gender": data["gender"].values,
                    "age": data["age"].values,
                    "occupation": data["occupation"].values}
        labels = data["item"].values
        train_data = tf.data.Dataset.from_tensor_slices((features, labels))
        return train_data.shuffle(10000).repeat(repeat).batch(batch_size).prefetch(buffer_size=1)

    elif mode == "eval":
        features = {"user": data["user"].values,
                    "gender": data["gender"].values,
                    "age": data["age"].values,
                    "occupation": data["occupation"].values}
        labels = data["item"].values
        eval_data = tf.data.Dataset.from_tensor_slices((features, labels))
        return eval_data.batch(batch_size)


def build_model(features, mode, params):
    use_bn = params["use_bn"]
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    net = fc.input_layer(features, params["feature_columns"])
    if use_bn:
        net = tf.layers.batch_normalization(net, training=is_training)

    for units in params["hidden_units"]:
        if use_bn:
            net = tf.layers.dense(net, units=units, activation=None, use_bias=False)
            net = tf.nn.relu(tf.layers.batch_normalization(net, training=is_training))
        else:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    if use_bn:
        net = tf.layers.dense(net, units=params["last_hidden_units"], activation=None, use_bias=False)
        net = tf.nn.relu(tf.layers.batch_normalization(net, training=is_training), name="user_vector_layer")
    else:
        net = tf.layers.dense(net, units=params["last_hidden_units"], activation=tf.nn.relu, name="user_vector_layer")

    return net


def model_fn(features, labels, mode, params):
    net = build_model(features, mode, params)
    nce_weights = tf.Variable(
        tf.truncated_normal([params["n_classes"], params["last_hidden_units"]],
                             stddev=1.0 / math.sqrt(params["last_hidden_units"])), name="nce_weights")
    nce_biases = tf.Variable(tf.zeros([params["n_classes"]]), name="nce_biases")
    logits = tf.matmul(net, tf.transpose(nce_weights)) + nce_biases
    top_k_values, top_k_indices = tf.nn.top_k(logits, params["top_k"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "user_vector": net,
            "top_k_values": top_k_values,
            "top_k_indices": top_k_indices}

        export_outputs = {"prediction": tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    labels = tf.reshape(labels, [-1, 1])
    if mode == tf.estimator.ModeKeys.EVAL:
        thresholds = params["eval_top_n"] if "eval_top_n" in params else [5, 10, 20, 50, 100]
        metrics = dict()
        for k in thresholds:
            metrics["recall/recall@" + str(k)] = tf.metrics.recall_at_k(labels, logits, int(k))
            metrics["precision/precision@" + str(k)] = tf.metrics.precision_at_k(labels, logits, int(k))
            correct = tf.nn.in_top_k(logits, tf.squeeze(labels), int(k))
            metrics["accuracy/accuracy@" + str(k)] = tf.metrics.accuracy(labels=tf.ones_like(labels, dtype=tf.float32),
                                                                         predictions=tf.to_float(correct))

        labels_one_hot = tf.one_hot(labels, params["n_classes"])
        labels_one_hot = tf.reshape(labels_one_hot, [-1, params["n_classes"]])
        print("labels_one_hot shape: ", labels_one_hot.get_shape())
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=labels,
        inputs=net,
        num_sampled=params["num_sampled"],
        num_classes=params["n_classes"],
        num_true=1,
        remove_accidental_hits=True,
        name="nce_loss"))

    optimizer = tf.train.AdamOptimizer(params["lr"])
    training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    training_op = tf.group([training_op, update_ops])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)


def main(unused_argv):
    set_tfconfig_environ()
    dataset = pd.read_csv(FLAGS.dataset, header=None, usecols=[0, 1, 3, 4, 5],
                          names=["user", "item", "gender", "age", "occupation"])
    item_unique = np.unique(dataset.item.values)
    print("num items: ", len(item_unique))
    item_id_map = dict(zip(item_unique, np.arange(len(item_unique))))
    dataset["item"] = dataset["item"].map(item_id_map)

    train_data, test_data = train_test_split(dataset)
    feature_columns = create_feature_columns(train_data)

    strategy = tf.distribute.experimental.ParameterServerStrategy()
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params={"feature_columns": feature_columns,
                "hidden_units": map(int, FLAGS.hidden_units.split(",")),
                "last_hidden_units": FLAGS.last_hidden_units,
                "lr": FLAGS.learning_rate,
                "use_bn": FLAGS.use_bn,
                "n_classes": FLAGS.n_classes,
                "num_sampled": FLAGS.num_sampled,
                "top_k": FLAGS.top_k,
                "eval_top_n": map(int, FLAGS.eval_top_n.split(","))},
        config=tf.estimator.RunConfig(model_dir="youtube_dir",
                                      save_checkpoints_steps=100000,
                                      train_distribute=strategy))

    print("train steps: ", FLAGS.train_steps, "batch size: ", FLAGS.batch_size)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_data, FLAGS.batch_size, mode="train"),
                                        max_steps=FLAGS.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test_data, FLAGS.batch_size, mode="eval"), steps=None)
    print("before train and evaluate")
    t0 = time.time()
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("after train and evaluate, training time: %.4f" % (time.time() - t0))

    t1 = time.time()
    results = classifier.evaluate(input_fn=lambda: input_fn(test_data, FLAGS.batch_size, mode="eval"))
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    print("after evaluate, evaluate time: %.4f" % (time.time() - t1))
    print("predict boolean: ", FLAGS.predict)

    if FLAGS.predict:
        pred = list(classifier.predict(input_fn=lambda: input_fn(test_data, FLAGS.batch_size, mode="eval")))
        import random
        random.shuffle(pred)
        print("pred result example: ")
        for i in range(2):
            print(pred[i])

    elif FLAGS.job_name == "worker" and FLAGS.task_index == 0:
        print("exporting model...")
        feature_spec = fc.make_parse_example_spec(feature_columns)
        print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        classifier.export_saved_model(FLAGS.saved_model_dir, serving_input_receiver_fn)

        print("save item vector...")
        nce_weights = classifier.get_variable_value("nce_weights")
        nce_biases = classifier.get_variable_value("nce_biases")
        [rows, cols] = nce_weights.shape
        with tf.gfile.FastGFile(FLAGS.output_item_vector, "w") as f:
            for i in range(rows):
                f.write(str(i) + "\t")
                for j in range(cols):
                    f.write(str(nce_weights[i, j]))
                    f.write(u",")
                f.write(str(nce_biases[i]))
                f.write(u"\n")
        print("quit main")



if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster:
        print("running cluster mode...")
        parse_argument()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.app.run(main=main)


# python youtube_distributed.py --dataset "./merged_data.csv" --saved_model_dir "./saved_model_dir" --hidden_units "128,64" --eval_top_n "10,50" --use_bn True --n_classes 3706 --train_steps 3706 --predict False --job_name "ps" --task_index 0 --run_on_cluster True



