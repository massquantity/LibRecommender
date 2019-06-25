import os
import time
import shutil
import logging
import numpy as np
import tensorflow as tf
from ..evaluate.evaluate import precision_tf
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class FmPure:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42, task="rating"):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task

    @staticmethod
    def build_sparse_data(data, user_indices, item_indices):
        first_dim = np.tile(np.arange(len(user_indices)), 2).reshape(-1, 1)
        second_dim = np.concatenate([user_indices, item_indices + data.n_users], axis=0).reshape(-1, 1)
        indices = np.concatenate([first_dim, second_dim], axis=1)
        indices = indices.astype(np.int64)
        values = np.ones(len(user_indices) * 2, dtype=np.float32)
        shape = [len(user_indices), data.n_users + data.n_items]
        return indices, values, shape

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.dim = dataset.n_users + dataset.n_items
        self.w = tf.Variable(tf.truncated_normal([self.dim, 1], 0.0, 0.01))
        self.v = tf.Variable(tf.truncated_normal([self.dim, self.n_factors], 0.0, 0.01))

        self.x = tf.sparse_placeholder(tf.float32, [None, self.dim])
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)

        self.pairwise_term = 0.5 * tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.v)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.v)))

        self.concat = tf.concat([self.linear_term, self.pairwise_term], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat, units=1, name="pred")
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

    def fit(self, dataset, verbose=1):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            if self.task == "rating":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = len(dataset.train_labels) // self.batch_size
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        user_batch = dataset.train_user_indices[n * self.batch_size: end]
                        item_batch = dataset.train_item_indices[n * self.batch_size: end]
                        label_batch = dataset.train_labels[n * self.batch_size: end]

                        indices_batch, values_batch, shape_batch = FmPure.build_sparse_data(dataset,
                                                                                            user_batch,
                                                                                            item_batch)
                        self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                                values_batch,
                                                                                                shape_batch),
                                                                   self.labels: label_batch})

                    if verbose > 0:
                        indices_train, values_train, shape_train = FmPure.build_sparse_data(
                                                                        dataset,
                                                                        dataset.train_user_indices,
                                                                        dataset.train_item_indices)
                        train_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_train,
                                                                                  values_train,
                                                                                  shape_train),
                                                                         self.labels: dataset.train_labels})

                        indices_test, values_test, shape_test = FmPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.test_user_indices,
                                                                    dataset.test_item_indices)
                        test_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_test,
                                                                                 values_test,
                                                                                 shape_test),
                                                                        self.labels: dataset.test_labels})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, train rmse: {:.4f}".format(epoch, train_rmse))
                        print("Epoch {}, test rmse: {:.4f}".format(epoch, test_rmse))
                        print()

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
                    n_batches = len(dataset.train_user_implicit) // self.batch_size
                    for n in range(n_batches):
                        user_batch, item_batch, label_batch = neg.next_batch()
                        indices_batch, values_batch, shape_batch = FmPure.build_sparse_data(dataset,
                                                                                            user_batch,
                                                                                            item_batch)
                        self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                                values_batch,
                                                                                                shape_batch),
                                                                   self.labels: label_batch})

                    if verbose > 0:
                        indices_train, values_train, shape_train = FmPure.build_sparse_data(
                                                                       dataset,
                                                                       dataset.train_user_implicit,
                                                                       dataset.train_item_implicit)
                        train_loss, train_accuracy, train_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.x: (indices_train, values_train, shape_train),
                                                     self.labels: dataset.train_label_implicit})

                        indices_test, values_test, shape_test = FmPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.test_user_implicit,
                                                                    dataset.test_item_implicit)
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.x: (indices_test, values_test, shape_test),
                                                     self.labels: dataset.test_label_implicit})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, train loss: {:.4f}, train accuracy: {:.4f}, train precision: {:.4f}".format(
                                epoch, train_loss, train_accuracy, train_precision))
                        print("Epoch {}, test loss: {:.4f}, test accuracy: {:.4f}, test precision: {:.4f}".format(
                                epoch, test_loss, test_accuracy, test_precision))
                        print()


    def predict(self, u, i):
        index, value, shape = FmPure.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape)})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred


class FmFeat:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42, task="rating"):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.w = tf.Variable(tf.truncated_normal([self.feature_size + 1, 1], 0.0, 0.01))  # feature_size + 1####
        self.v = tf.Variable(tf.truncated_normal([self.feature_size + 1, self.n_factors], 0.0, 0.01))
        self.feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        self.linear_embedding = tf.nn.embedding_lookup(self.w, self.feature_indices)   # N * F * 1
        self.linear_term = tf.reduce_sum(tf.multiply(self.linear_embedding, self.feature_values_reshape), 2)

        self.feature_embedding = tf.nn.embedding_lookup(self.v, self.feature_indices)  # N * F * K
        self.feature_embedding = tf.multiply(self.feature_embedding, self.feature_values_reshape)

        self.pairwise_term = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(self.feature_embedding, axis=1)),
            tf.reduce_sum(tf.square(self.feature_embedding), axis=1))

        self.concat = tf.concat([self.linear_term, self.pairwise_term], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat, units=1)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])  # reg_w

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

    def fit(self, dataset, verbose=1, pre_sampling=True):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            if self.task == "rating":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = len(dataset.train_labels) // self.batch_size
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        indices_batch = dataset.train_feat_indices[n * self.batch_size: end]
                        values_batch = dataset.train_feat_values[n * self.batch_size: end]
                        labels_batch = dataset.train_labels[n * self.batch_size: end]

                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                    if verbose > 0:
               #         train_rmse = self.rmse.eval(feed_dict={self.feature_indices: dataset.train_feat_indices,
                #                                               self.feature_values: dataset.train_feat_values,
                #                                               self.labels: dataset.train_labels})

                        test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                              feed_dict={
                                                                  self.feature_indices: dataset.test_feat_indices,
                                                                  self.feature_values: dataset.test_feat_values,
                                                                  self.labels: dataset.test_labels})

                #        print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                #                epoch, train_rmse, time.time() - t0))
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))
                        print()

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSamplingFeat(dataset, dataset.num_neg, self.batch_size, pre_sampling=pre_sampling)
                    n_batches = len(dataset.train_indices_implicit) // self.batch_size
                    for n in range(n_batches):
                        indices_batch, values_batch, labels_batch = neg.next_batch()
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                #        pred, logits, loss, acc, pre = self.sess.run([self.pred, self.logits, self.loss, self.accuracy, self.precision],
                #                                       feed_dict={self.feature_indices: indices_batch,
                #                                                   self.feature_values: values_batch,
                #                                                   self.labels: labels_batch})
                #        if n % 100 == 0:
                        #    print("mm: ", pred[:10], loss, acc, pre)
                #            print("batch pred: ", pred[:10])
                #            print("batch labe: ", labels_batch[:10])
                #            print("accuracy: ", acc, pre)

                    if verbose > 0:
                #        train_loss, train_accuracy, train_precision = \
                #            self.sess.run([self.loss, self.accuracy, self.precision],
                #                          feed_dict={self.feature_indices: dataset.train_indices_implicit,
                #                                     self.feature_values: dataset.train_values_implicit,
                #                                     self.labels: dataset.train_labels_implicit})

                        test_pred, test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.pred, self.loss, self.accuracy, self.precision],
                                          feed_dict={self.feature_indices: dataset.test_indices_implicit,
                                                     self.feature_values: dataset.test_values_implicit,
                                                     self.labels: dataset.test_labels_implicit})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                #        print("Epoch {}, train loss: {:.4f}, train accuracy: {:.4f}, train precision: {:.4f}".format(
                #            epoch, train_loss, train_accuracy, train_precision))
                        print("Epoch {}, test loss: {:.4f}, test accuracy: {:.4f}, test precision: {:.4f}".format(
                            epoch, test_loss, test_accuracy, test_precision))
                        print()

    def predict(self, feat_ind, feat_val):
        try:
            pred = self.sess.run(self.pred, feed_dict={self.feature_indices: feat_ind,
                                                       self.feature_values: feat_val})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred

    def export_model(self, version, simple_save=False):
        model_base_path = os.path.realpath(".")
        export_path = os.path.join(model_base_path, "serving", "FM", version)
        if os.path.isdir(export_path):
            logging.warning("\tModel path \"%s\" already exists, removing..." % export_path)
            shutil.rmtree(export_path)
        if simple_save:
            print("simple_save is deprecated, it will be removed in tensorflow xxx...")
            tf.saved_model.simple_save(self.sess, export_path,
                                       inputs={'fi': self.feature_indices,
                                               'fv': self.feature_values},
                                       outputs={'y_prob': self.y_prob})
        else:
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            input_fi = tf.saved_model.utils.build_tensor_info(self.feature_indices)
            input_fv = tf.saved_model.utils.build_tensor_info(self.feature_values)
        #    input_label = tf.saved_model.utils.build_tensor_info(self.labels)
            input_y = tf.saved_model.utils.build_tensor_info(self.y_prob)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'fi': input_fi,
                            'fv': input_fv},
                    outputs={'y_prob': input_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                self.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={'predict': prediction_signature}
            # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            #    main_op=tf.tables_initializer(),
            #    strip_default_attrs=True
            )

            builder.save()
        logging.warning('\tDone exporting!')
