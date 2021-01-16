import abc
import inspect
import json
import os
import multiprocessing
import time
import numpy as np
import tensorflow as tf2
from ..utils.misc import time_block, colorize
from ..utils.exception import NotSamplingError
tf = tf2.compat.v1
tf.disable_v2_behavior()


class Base(abc.ABC):
    """Base class for all recommendation models.

    Parameters
    ----------
    task : str
        Specific task, either rating or ranking.
    data_info : `DataInfo` object
        Object that contains useful information for training and predicting.
    lower_upper_bound : list or tuple, optional
        Lower and upper score bound for rating task.
    """

    def __init__(self, task, data_info, lower_upper_bound=None):
        self.task = task
        if task == "rating":
            self.global_mean = data_info.global_mean
            if lower_upper_bound is not None:
                assert isinstance(lower_upper_bound, (list, tuple)), (
                    "must contain both lower and upper bound if provided")
                self.lower_bound = lower_upper_bound[0]
                self.upper_bound = lower_upper_bound[1]
            else:
                self.lower_bound, self.upper_bound = data_info.min_max_rating
        #    print(f"lower bound: {self.lower_bound}, "
        #          f"upper bound: {self.upper_bound}")

        elif task != "ranking":
            raise ValueError("task must either be rating or ranking")

        self.default_prediction = (
            data_info.global_mean
            if task == "rating"
            else 0.0
        )

    @abc.abstractmethod
    def fit(self, train_data, **kwargs):
        """Train model on the training data.

        Parameters
        ----------
        train_data : `TransformedSet` object
            Data object used for training.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, user, item, **kwargs):
        """Predict score for given user and item.

        Parameters
        ----------
        user : int or array_like
            User id or batch of user ids.
        item : int or array_like
            Item id or batch of item ids.

        Returns
        -------
        prediction : int or array_like
            Predicted scores for each user-item pair.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recommend_user(self, user, n_rec, **kwargs):
        """Recommend a list of items for given user.

        Parameters
        ----------
        user : int
            User id to recommend.
        n_rec : int
            number of recommendations to return.

        Returns
        -------
        result : list of tuples
            A recommendation list, each recommendation
            contains an (item_id, score) tuple.

        """
        raise NotImplementedError

    def convert_id(self, user, item, inner_id=False):
        if not inner_id:
            user = (
                [self.data_info.user2id[user]]
                if isinstance(user, int)
                else [self.data_info.user2id[u] for u in user]
            )
            item = (
                [self.data_info.item2id[item]]
                if isinstance(item, int)
                else [self.data_info.item2id[i] for i in item]
            )
        else:
            user = [user] if isinstance(user, int) else user
            item = [item] if isinstance(item, int) else item
        return  np.asarray(user), np.asarray(item)

    def _check_unknown(self, user, item):
        unknown_user_indices = list(
            np.where(np.logical_or(user >= self.n_users, user < 0))[0]
        )
        unknown_item_indices = list(
            np.where(np.logical_or(item >= self.n_items, item < 0))[0]
        )

        unknown_user = (list(user[unknown_user_indices])
                        if unknown_user_indices
                        else None)
        unknown_item = (list(item[unknown_item_indices])
                        if unknown_item_indices
                        else None)
        unknown_index = list(
            set(unknown_user_indices) | set(unknown_item_indices)
        )
        unknown_num = len(unknown_index)

        if unknown_num > 0:
            # temp conversion, will convert back in the main model
            user[unknown_index] = 0
            item[unknown_index] = 0
            unknown_str = (f"Detect {unknown_num} unknown interaction(s), "
                           f"including user: {unknown_user}, "
                           f"item: {unknown_item}, "
                           f"will be handled as default prediction")
            print(f"{colorize(unknown_str, 'red')}")
        return unknown_num, unknown_index, user, item

    def _check_unknown_user(self, user):
        if 0 <= user < self.n_users:
            return user
        else:
            unknown_str = (f"detect unknown user {user}, "
                           f"return default recommendation")
            print(f"{colorize(unknown_str, 'red')}")
            return

    @staticmethod
    def _check_has_sampled(data, verbose):
        if not data.has_sampled and verbose > 1:
            exception_str = (f"During training, "
                             f"one must do whole data sampling "
                             f"before evaluating on epochs.")
            raise NotSamplingError(f"{colorize(exception_str, 'red')}")

    @staticmethod
    def _check_interaction_mode(recent_num, random_num):
        if recent_num is not None:
            assert isinstance(recent_num, int), "recent_num must be integer"
            mode = "recent"
            num = recent_num
        elif random_num is not None:
            assert isinstance(random_num, int), "random_num must be integer"
            mode = "random"
            num = random_num
        else:
            mode = "recent"
            num = 10  # by default choose 10 recent interactions
        return mode, num

    @staticmethod
    def _decide_sparse_indices(data_info):
        return False if not data_info.sparse_col.name else True

    @staticmethod
    def _decide_dense_values(data_info):
        return False if not data_info.dense_col.name else True

    @staticmethod
    def _sparse_feat_size(data_info):
        if (data_info.user_sparse_unique is not None
                and data_info.item_sparse_unique is not None):
            return max(np.max(data_info.user_sparse_unique),
                       np.max(data_info.item_sparse_unique)) + 1
        elif data_info.user_sparse_unique is not None:
            return np.max(data_info.user_sparse_unique) + 1
        elif data_info.item_sparse_unique is not None:
            return np.max(data_info.item_sparse_unique) + 1

    @staticmethod
    def _sparse_field_size(data_info):
        return len(data_info.sparse_col.name)

    @staticmethod
    def _dense_field_size(data_info):
        return len(data_info.dense_col.name)

    @staticmethod
    def show_start_time():
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"Training start time: {colorize(start_time, 'magenta')}")

    def save_params(self, path):
        hparams = dict()
        arg_names = list(inspect.signature(self.__init__).parameters.keys())
        arg_names.remove("data_info")
        for p in arg_names:
            hparams[p] = self.all_args[p]

        param_path = os.path.join(path, "hyper_parameters.json")
        with open(param_path, 'w') as f:
            json.dump(hparams, f, separators=(',', ':'))

    @classmethod
    def load_params(cls, path, data_info):
        if not os.path.exists(path):
            raise OSError(f"file folder {path} doesn't exists...")

        param_path = os.path.join(path, "hyper_parameters.json")
        with open(param_path, 'r') as f:
            hparams = json.load(f)
        hparams.update({"data_info": data_info})
        return hparams


class TfMixin(object):
    def __init__(self, tf_sess_config=None):
        self.cpu_num = multiprocessing.cpu_count()
        self.sess = self._sess_config(tf_sess_config)

    def _sess_config(self, tf_sess_config=None):
        if not tf_sess_config:
            # Session config based on:
            # https://software.intel.com/content/www/us/en/develop/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus.html
            tf_sess_config = {
                "intra_op_parallelism_threads": 0,
                "inter_op_parallelism_threads": 0,
                "allow_soft_placement": True,
                "device_count": {"CPU": self.cpu_num}
            }
        #    os.environ["OMP_NUM_THREADS"] = f"{self.cpu_num}"

        config = tf.ConfigProto(**tf_sess_config)
        return tf.Session(config=config)

    def train_pure(self, data_generator, verbose, shuffle, eval_data, metrics,
                   **kwargs):
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for user, item, label, _, _ in data_generator(
                        shuffle, self.batch_size
                ):
                    feed_dict = {self.user_indices: user,
                                 self.item_indices: item,
                                 self.labels: label}
                    if hasattr(self, "is_training"):
                        feed_dict.update({self.is_training: True})

                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict=feed_dict)

                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")

                class_name = self.__class__.__name__.lower()
                if class_name.startswith("svd"):
                    # set up parameters for prediction evaluate
                    self._set_latent_factors()

                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("="*30)

    def train_feat(self, data_generator, verbose, shuffle, eval_data, metrics,
                   **kwargs):
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(f"With lr_decay, epoch {epoch} learning rate: "
                      f"{self.sess.run(self.lr)}")
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for u, i, label, si, dv in data_generator(
                        shuffle, self.batch_size
                ):
                    feed_dict = self._get_feed_dict(u, i, si, dv, label, True)
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict)
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("="*30)

    def train_seq(self):
        pass  # TODO: combine train_feat and train_seq

    def _get_feed_dict(self, user_indices, item_indices, sparse_indices,
                       dense_values, label, is_training):
        feed_dict = {
            self.user_indices: user_indices,
            self.item_indices: item_indices,
            self.is_training: is_training
        }
        if self.sparse:
            feed_dict.update({self.sparse_indices: sparse_indices})
        if self.dense:
            feed_dict.update({self.dense_values: dense_values})
        if label is not None:
            feed_dict.update({self.labels: label})
        return feed_dict

    def _get_seq_feed_dict(self, u_interacted_seq, u_interacted_len,
                           user_indices, item_indices, label, sparse_indices,
                           dense_values, is_training):
        feed_dict = {
            self.user_interacted_seq: u_interacted_seq,
            self.user_interacted_len: u_interacted_len,
            self.user_indices: user_indices,
            self.item_indices: item_indices,
            self.is_training: is_training
        }
        if self.sparse:
            feed_dict.update({self.sparse_indices: sparse_indices})
        if self.dense:
            feed_dict.update({self.dense_values: dense_values})
        if label is not None:
            feed_dict.update({self.labels: label})
        return feed_dict

    def assign_oov(self):
        assign_ops = []
        for v in tf.trainable_variables():
            if hasattr(self, "user_variables"):
                for vu in self.user_variables:
                    if v.name.startswith(vu):
                        size = v.get_shape().as_list()[1]
                        zero_op = tf.IndexedSlices(
                            tf.zeros([1, size], dtype=tf.float32),
                            [self.n_users]
                        )
                        assign_ops.append(v.scatter_update(zero_op))
            if hasattr(self, "item_variables"):
                for vi in self.item_variables:
                    if v.name.startswith(vi):
                        size = v.get_shape().as_list()[1]
                        zero_op = tf.IndexedSlices(
                            tf.zeros([1, size], dtype=tf.float32),
                            [self.n_items]
                        )
                        assign_ops.append(v.scatter_update(zero_op))
        self.sess.run(assign_ops)

    def save_tf_model(self, path, model_name):
        model_path = os.path.join(path,  model_name)
        saver = tf.train.Saver()
        saver.save(self.sess, model_path, write_meta_graph=True)

    @classmethod
    def load_tf_model(cls, path, model_name, data_info):
        model_path = os.path.join(path, model_name)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model._build_model()
        if hasattr(model, "user_last_interacted"):
            model._set_last_interacted()
        # saver = tf.train.import_meta_graph(os.path.join(path, model_name + ".meta"))
        saver = tf.train.Saver()
        saver.restore(model.sess, model_path)
        return model

    def save_variables(self, path, model_name):
        variable_path = os.path.join(path, f"{model_name}_variables")
        variables = dict()
        for v in tf.global_variables():
            variables[v.name] = self.sess.run(v)
        np.savez_compressed(variable_path, **variables)

    @classmethod
    def load_variables(cls, path, model_name, data_info):
        variable_path = os.path.join(path, f"{model_name}_variables.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model._build_model()
        if hasattr(model, "user_last_interacted"):
            model._set_last_interacted()
        # model.sess.run(tf.trainable_variables()[0].initializer)
        # print(model.sess.run(tf.trainable_variables()[0]))
        assign_ops = []
        for v in tf.global_variables():
            assign_ops.append(v.assign(variables[v.name]))
            # v.load(variables[v.name], session=model.sess)
        model.sess.run(assign_ops)
        # print(model.sess.run(tf.trainable_variables()[0]))
        return model
