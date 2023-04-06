Model Retrain
=============

The Problem
-----------

When we get some new data, we definitely want to retrain the old model with these new data,
but this is actually not easy for some deep learning models.

The reason lies in the new users/items that may appear in the new data.
In deep learning models, embedding variables are commonly used, and their shapes are preassigned and fixed during and after training.
The same issue also goes with new features. If we load the old model and want to train it with new users/items/features,
these embedding shapes must be expanded, which is not allowed in TensorFlow and PyTorch (Well, at least to my knowledge).

One workaround is to combine the new data with the old data, then retrain the model with all the data.
But it would be a waste of time and resources to retrain on the whole data every time we get some new data.

Bigger Problem?
---------------

So how can we retrain a model if we can't change the shape of the variables in TensorFlow and PyTorch?
Well, if we can't alter it, we create a new one, then explicitly assign the old one to the new one.
Specifically, in TensorFlow, we build a new graph with variables with new shapes,
then assign the old values to the correct indices of new variables. For the new indices,
i.e. the new user/item part, they are initialized randomly as usual.

The problem with this solution is that we can not use TensorFlow's default method such as
`tf.train.Saver <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver>`_ or
`tf.saved_model <https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/saved_model>`_,
as well as PyTorch's default method such as `load_state_dict <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict>`_
, since these can only load to the exact same model with same shapes.

Things become even more desperate if we also want to save and restore the optimizers' variables,
e.g. the first and second moments in Adam optimizer. Since these variables are used as states in optimizers,
failing to keep them means losing the previous training state.

Solution in LibRecommender
--------------------------

So our solution is extracting all the variables to :class:`numpy.ndarray` format, then saving them using the ``save``
method in numpy. After that the variables are loaded from numpy, we then build the new graph and
update the new variables with old ones.

So it's crucial to set ``manual=True, inference_only=False`` when you save the model, which means
leveraging the numpy way. If you set ``manual=False``, the model may use the ``tf.train.Saver`` or
`torch.save <https://pytorch.org/docs/stable/generated/torch.save.html>`_ to save
the model, which is OK if you are certain that there will be no new user/item in new data.

.. literalinclude:: ../../../examples/model_retrain_example.py
   :caption: From file `examples/model_retrain_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/model_retrain_example.py>`_
   :name: model_retrain_example.py
   :lines: 71-73

Before retraining the model, the new data should also be transformed. Since the old ``data_info``
already exists, we need to merge new information with the old one,
especially those new users/items/features from new data. This is achieved by calling
:meth:`~libreco.data.dataset.DatasetFeat.merge_trainset`,
:meth:`~libreco.data.dataset.DatasetFeat.merge_evalset`,
:meth:`~libreco.data.dataset.DatasetFeat.merge_testset`
functions.

During recommendation, we usually want to filter some items that a user has previously consumed,
which are also stored in :class:`~libreco.data.DataInfo` object. So if you want to combine the user-consumed
information in old data with that in new data, you can pass ``merge_behavior=True``:

.. _retrain_data:

.. code-block:: python3

   >>> train_data, new_data_info = DatasetFeat.merge_trainset(train, data_info, merge_behavior=True)

Finally, loading the old variables and assigning them to the new model requires only one function :meth:`~libreco.algorithms.DeepFM.rebuild_model`:

.. code-block:: python3

   >>> model.rebuild_model(path="model_path", model_name="deepfm_model", full_assign=True)

.. SeeAlso::

    `model_retrain_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/model_retrain_example.py>`__
