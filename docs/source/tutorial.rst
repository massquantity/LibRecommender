Tutorial
========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/massquantity/LibRecommender/blob/master/examples/tutorial.ipynb
   :alt: Open in Colab

.. image:: https://img.shields.io/badge/View-on%20GitHub-blue?logo=GitHub
   :target: https://github.com/massquantity/LibRecommender/blob/master/examples/tutorial.ipynb
   :alt: View On GitHub

This tutorial will walk you through the comprehensive process of
training a model in LibRecommender, i.e. **data processing -> feature
engineering -> training -> evaluate -> save/load -> retrain**. We will
use `Wide & Deep <https://arxiv.org/pdf/1606.07792.pdf>`__ as the
example algorithm.

First make sure the latest LibRecommender has been installed:

.. code-block:: bash

    $ pip install -U LibRecommender

.. admonition:: Serving
    :class: Note

    For how to deploy a trained model in LibRecommender, see :ref:`Serving Guide <Serving Guide>`.

.. admonition:: TensorFlow1 issue
    :class: Error

    If you encounter errors like
    ``Variables already exist, disallowed...`` in this tutorial, just call
    ``tf.compat.v1.reset_default_graph()`` first. It's one of the inconvenience from TensorFlow1.



Load Data
---------

In this tutorial we will use the `MovieLens
1M <https://grouplens.org/datasets/movielens/1m/>`__ dataset. The
following code will load the data into ``pandas.DataFrame`` format. If
the data does not exist locally, it will be downloaded at first.

.. code:: python3

    import random
    import warnings
    import zipfile
    from pathlib import Path

    import pandas as pd
    import tensorflow as tf
    import tqdm
    warnings.filterwarnings("ignore")

.. code:: python3

    def load_ml_1m():
        # download and extract zip file
        tf.keras.utils.get_file(
            "ml-1m.zip",
            "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
            cache_dir=".",
            cache_subdir=".",
            extract=True,
        )
        # read and merge data into same table
        cur_path = Path(".").absolute()
        ratings = pd.read_csv(
            cur_path / "ml-1m" / "ratings.dat",
            sep="::",
            usecols=[0, 1, 2, 3],
            names=["user", "item", "rating", "time"],
        )
        users = pd.read_csv(
            cur_path / "ml-1m" / "users.dat",
            sep="::",
            usecols=[0, 1, 2, 3],
            names=["user", "sex", "age", "occupation"],
        )
        items = pd.read_csv(
            cur_path / "ml-1m" / "movies.dat",
            sep="::",
            usecols=[0, 2],
            names=["item", "genre"],
            encoding="iso-8859-1",
        )
        items[["genre1", "genre2", "genre3"]] = (
            items["genre"].str.split(r"|", expand=True).fillna("missing").iloc[:, :3]
        )
        items.drop("genre", axis=1, inplace=True)
        data = ratings.merge(users, on="user").merge(items, on="item")
        data.rename(columns={"rating": "label"}, inplace=True)
        # random shuffle data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        return data

.. code:: python3

    >>> data = load_ml_1m()
    >>> data.shape


.. parsed-literal::

    data shape: (1000209, 10)


.. code:: python3

    >>> data.iloc[random.choices(range(len(data)), k=10)]  # randomly select 10 rows


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>user</th>
          <th>item</th>
          <th>label</th>
          <th>time</th>
          <th>sex</th>
          <th>age</th>
          <th>occupation</th>
          <th>genre1</th>
          <th>genre2</th>
          <th>genre3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>951319</th>
          <td>4913</td>
          <td>3538</td>
          <td>3</td>
          <td>962677962</td>
          <td>F</td>
          <td>25</td>
          <td>1</td>
          <td>Comedy</td>
          <td>missing</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>969300</th>
          <td>3246</td>
          <td>2977</td>
          <td>5</td>
          <td>968309625</td>
          <td>F</td>
          <td>35</td>
          <td>1</td>
          <td>Comedy</td>
          <td>Drama</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>914441</th>
          <td>1181</td>
          <td>3015</td>
          <td>2</td>
          <td>976142934</td>
          <td>M</td>
          <td>35</td>
          <td>7</td>
          <td>Thriller</td>
          <td>missing</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>905593</th>
          <td>2063</td>
          <td>695</td>
          <td>2</td>
          <td>974665086</td>
          <td>M</td>
          <td>25</td>
          <td>4</td>
          <td>Mystery</td>
          <td>Thriller</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>512570</th>
          <td>4867</td>
          <td>1200</td>
          <td>4</td>
          <td>962817971</td>
          <td>M</td>
          <td>25</td>
          <td>16</td>
          <td>missing</td>
          <td>missing</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>524227</th>
          <td>4684</td>
          <td>3174</td>
          <td>2</td>
          <td>963667810</td>
          <td>F</td>
          <td>25</td>
          <td>0</td>
          <td>Comedy</td>
          <td>Drama</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>801408</th>
          <td>3792</td>
          <td>1224</td>
          <td>4</td>
          <td>966360592</td>
          <td>M</td>
          <td>25</td>
          <td>6</td>
          <td>Drama</td>
          <td>War</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>117662</th>
          <td>2270</td>
          <td>480</td>
          <td>5</td>
          <td>974574449</td>
          <td>M</td>
          <td>18</td>
          <td>1</td>
          <td>Action</td>
          <td>Adventure</td>
          <td>Sci-Fi</td>
        </tr>
        <tr>
          <th>935170</th>
          <td>1088</td>
          <td>3825</td>
          <td>1</td>
          <td>1037975844</td>
          <td>F</td>
          <td>1</td>
          <td>10</td>
          <td>Drama</td>
          <td>missing</td>
          <td>missing</td>
        </tr>
        <tr>
          <th>309994</th>
          <td>4808</td>
          <td>3051</td>
          <td>3</td>
          <td>962934115</td>
          <td>M</td>
          <td>35</td>
          <td>0</td>
          <td>Drama</td>
          <td>missing</td>
          <td>missing</td>
        </tr>
      </tbody>
    </table>
    </div>



Now we have about 1 million data. In order to perform evaluation after training, we need to split the data
into train, eval and test data first. In this tutorial we will simply
use :meth:`~libreco.data.random_split`. For other ways of splitting data, see :doc:`user_guide/data_processing`.

.. _two parts:

.. NOTE::

   For now, We will only use **first half data** for training. Later we will use the rest data to retrain the model.


Process Data & Features
-----------------------

.. code:: python3

    >>> from libreco.data import random_split
    
    # split data into three folds for training, evaluating and testing
    >>> first_half_data = data[: (len(data) // 2)]
    >>> train_data, eval_data, test_data = random_split(first_half_data, multi_ratios=[0.8, 0.1, 0.1], seed=42)

.. code:: python3

    >>> print("first half data shape:", first_half_data.shape)

::

    first half data shape: (500104, 10)

The data contains some categorical features such as “sex” and “genre”,
as well as a numerical feature “age”. In LibRecommender we use
``sparse_col`` to represent categorical features and ``dense_col`` to
represent numerical features. So one should specify the column
information and then use ``DatasetFeat.build_*`` functions to process
the data.

.. code:: python3

    >>> from libreco.data import DatasetFeat
    
    >>> sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    >>> dense_col = ["age"]
    >>> user_col = ["sex", "age", "occupation"]
    >>> item_col = ["genre1", "genre2", "genre3"]
    
    >>> train_data, data_info = DatasetFeat.build_trainset(train_data, user_col, item_col, sparse_col, dense_col)
    >>> eval_data = DatasetFeat.build_evalset(eval_data)
    >>> test_data = DatasetFeat.build_testset(test_data)

``user_col`` means features belong to user, and ``item_col`` means features
belong to item. Note that the column numbers should match,
i.e. ``len(sparse_col) + len(dense_col) == len(user_col) + len(item_col)``.

.. code:: python3

    >>> print(data_info)


.. parsed-literal::

    n_users: 6040, n_items: 3576, data density: 1.8523 %


Training the Model
------------------

Now with all the data and features prepared, we can start training the
model!

Since as its name suggests, the ``Wide & Deep`` algorithm has wide and
deep parts, and they use different optimizers. So we should specify the
learning rate separately by using a dict:
``{"wide": 0.01, "deep": 3e-4}``. For other model hyper-parameters, see API reference of
:class:`~libreco.algorithms.WideDeep`.

In this example we treat all the samples in data as positive samples,
and perform negative sampling. This is a standard procedure for "implicit data".

.. code:: python3

    from libreco.algorithms import WideDeep

.. code:: python3

    model = WideDeep(
        task="ranking",
        data_info=data_info,
        embed_size=16,
        n_epochs=2,
        loss_type="cross_entropy",
        lr={"wide": 0.05, "deep": 7e-4},
        batch_size=2048,
        use_bn=True,
        hidden_units=(128, 64, 32),
    )

    model.fit(
        train_data,
        neg_sampling=True,  # perform negative sampling on training and eval data
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

::

    Epoch 1 elapsed: 2.905s
        train_loss: 0.959
        eval log_loss: 0.5823
        eval roc_auc: 0.8032
        eval precision@10: 0.0236
        eval recall@10: 0.0339
        eval ndcg@10: 0.1001

    Epoch 2 elapsed: 2.508s
        train_loss: 0.499
        eval log_loss: 0.4769
        eval roc_auc: 0.8488
        eval precision@10: 0.0332
        eval recall@10: 0.0523
        eval ndcg@10: 0.1376

We’ve trained the model for 2 epochs and evaluated the performance on
the eval data during training. Next we can evaluate on the *independent*
test data.

.. code:: python3

    >>> from libreco.evaluation import evaluate
    >>> evaluate(
    >>>     model=model,
    >>>     data=test_data,
    >>>     neg_sampling=True,  # perform negative sampling on test data
    >>>     metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    >>> )

.. parsed-literal::

    {'loss': 0.4782908669403157,
     'roc_auc': 0.8483713737644527,
     'precision': 0.031268748897123694,
     'recall': 0.04829594849021039,
     'ndcg': 0.12866793895121623}



Make Recommendation
-------------------

The recommend part is pretty straightforward. You can make
recommendation for one user or a batch of users.

.. code:: python3

    >>> model.recommend_user(user=1, n_rec=3)

.. parsed-literal::

    {1: array([ 364, 3751, 2858])}


.. code:: python3

    >>> model.recommend_user(user=[1, 2, 3], n_rec=3)

.. parsed-literal::

    {1: array([ 364, 3751, 2858]),
     2: array([1617,  608,  912]),
     3: array([ 589, 2571, 1200])}

You can also make recommdation based on specific user features.

.. code:: python3

    >>> model.recommend_user(user=1, n_rec=3, user_feats={"sex": "M", "age": 33})

.. parsed-literal::

    {1: array([2716,  589, 2571])}


.. code:: python3

    >>> model.recommend_user(user=1, n_rec=3, user_feats={"occupation": 17})

.. parsed-literal::

    {1: array([2858, 1210, 1580])}



Save, Load and Inference
------------------------

When saving the model, we should also save the ``data_info`` for feature
information.

.. code:: python3

    >>> data_info.save("model_path", model_name="wide_deep")
    >>> model.save("model_path", model_name="wide_deep")

Then we can load the model and make recommendation again.

.. code:: python3

    >>> tf.compat.v1.reset_default_graph()  # need to reset graph in TensorFlow1

.. code:: python3

    >>> from libreco.data import DataInfo
    
    >>> loaded_data_info = DataInfo.load("model_path", model_name="wide_deep")
    >>> loaded_model = WideDeep.load("model_path", model_name="wide_deep", data_info=loaded_data_info)
    >>> loaded_model.recommend_user(user=1, n_rec=3)


Retrain the Model with New Data
-------------------------------

Remember that we split the original ``MovieLens 1M`` data into :ref:`two parts <two parts>`
in the first place? We will treat the **second half** of the data as our
new data and retrain the saved model with it. In real-world recommender
systems, data may be generated every day, so it is inefficient to train
the model from scratch every time we get some new data.

.. code:: python3

    >>> second_half_data = data[(len(data) // 2) :]
    >>> train_data, eval_data = random_split(second_half_data, multi_ratios=[0.8, 0.2])

.. code:: python3

    >>> print("second half data shape:", second_half_data.shape)

::

    second half data shape: (500105, 10)


The data processing is similar, except that we should use
:meth:`~libreco.data.dataset.DatasetFeat.merge_trainset` and :meth:`~libreco.data.dataset.DatasetFeat.merge_evalset`
in :class:`~libreco.data.dataset.DatasetFeat`.

The purpose of these functions is combining information from old data
with that from new data, especially for the possible new users/items
from new data. For more details, see :doc:`user_guide/model_retrain`.

.. code:: python3

    >>> # pass `loaded_data_info` and get `new_data_info`
    >>> train_data, new_data_info = DatasetFeat.merge_trainset(train_data, loaded_data_info, merge_behavior=True)
    >>> eval_data = DatasetFeat.merge_evalset(eval_data, new_data_info)  # use new_data_info

Then we construct a new model, and call :meth:`~libreco.algorithms.WideDeep.rebuild_model`
method to assign the old trained variables into the new model.

.. code:: python3

    >>> tf.compat.v1.reset_default_graph()  # need to reset graph in TensorFlow1

.. code:: python3

    new_model = WideDeep(
        task="ranking", 
        data_info=new_data_info,  # pass new_data_info
        embed_size=16, 
        n_epochs=2,
        loss_type="cross_entropy",
        lr={"wide": 0.01, "deep": 1e-4},
        batch_size=2048,
        use_bn=True,
        hidden_units=(128, 64, 32), 
    )
    
    new_model.rebuild_model(path="model_path", model_name="wide_deep", full_assign=True)

Finally, the training and recommendation parts are the same as before.

.. code:: python3

    new_model.fit(
        train_data,
        neg_sampling=True,
        verbose=2, 
        shuffle=True, 
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

::

    Epoch 1 elapsed: 2.867s
        train_loss: 0.4867
        eval log_loss: 0.4482
        eval roc_auc: 0.8708
        eval precision@10: 0.0985
        eval recall@10: 0.0710
        eval ndcg@10: 0.3062

    Epoch 2 elapsed: 2.770s
        train_loss: 0.472
        eval log_loss: 0.4416
        eval roc_auc: 0.8741
        eval precision@10: 0.1031
        eval recall@10: 0.0738
        eval ndcg@10: 0.3168


.. code:: python3

    >>> new_model.recommend_user(user=1, n_rec=3)

.. parsed-literal::

    {1: array([ 364, 2858, 1210])}

.. code:: python3

    >>> new_model.recommend_user(user=[1, 2, 3], n_rec=3)

.. parsed-literal::

    {1: array([ 364, 2858, 1210]),
     2: array([ 608, 1617, 1233]),
     3: array([ 589, 2571, 1387])}

**This completes our tutorial!**

.. admonition:: Where to go from here
    :class: Note

    For more examples, see the `examples/ <https://github.com/massquantity/LibRecommender/tree/master/examples>`__ folder on GitHub.

    For more usages, please head to :doc:`user_guide/index`.

    For serving a trained model, please head to :doc:`serving_guide/python`.
