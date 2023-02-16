Tutorial
========

This tutorial will walk you through the comprehensive process of
training a model in LibRecommender, i.e. **data processing -> feature
engineering -> training -> evaluate -> save/load -> retrain**. We will
use `Wide & Deep <https://arxiv.org/pdf/1606.07792.pdf>`__ as the
example algorithm.

First make sure LibRecommender is installed.

.. code-block:: bash

    $ pip install LibRecommender

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
    from urllib.request import urlretrieve
    
    import pandas as pd
    import tensorflow as tf
    import tqdm
    warnings.filterwarnings("ignore")

.. code:: python3

    def split_genre(line):
        genres = line.split("|")
        if len(genres) == 3:
            return genres[0], genres[1], genres[2]
        elif len(genres) == 2:
            return genres[0], genres[1], "missing"
        elif len(genres) == 1:
            return genres[0], "missing", "missing"
        else:
            return "missing", "missing", "missing"

.. code:: python3

    def load_ml_1m():
        download_path = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        original_file = "ml-1m.zip"
        cur_path = Path(".").absolute()
        if not Path.exists(Path(original_file)):
            print("Data does not exist, start downloading...")
            with tqdm.tqdm(unit='B', unit_scale=True) as p:
                def report(chunk, chunksize, total):
                    p.total = total
                    p.update(chunksize)
                urlretrieve(download_path, original_file, reporthook=report)
            print("Download successful!")
        # extract zip file
        with zipfile.ZipFile(original_file, 'r') as f:
            f.extractall(cur_path)
    
        # read and merge data into same table
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
        items["genre1"], items["genre2"], items["genre3"] = zip(*items["genre"].apply(split_genre))
        items.drop("genre", axis=1, inplace=True)
        data = ratings.merge(users, on="user").merge(items, on="item")
        data.rename(columns={"rating": "label"}, inplace=True)
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

In this example we treat all the samples in data as positive samples,
and perform negative sampling. This is a standard procedure for "implicit data".

.. code:: python3

    # sample negative items for each record
    >>> train_data.build_negative_samples(data_info)
    >>> eval_data.build_negative_samples(data_info)
    >>> test_data.build_negative_samples(data_info)


Training the Model
------------------

Now with all the data and features prepared, we can start training the
model!

Since as its name suggests, the ``Wide & Deep`` algorithm has wide and
deep parts, and they use different optimizers. So we should specify the
learning rate separately by using a dict:
``{"wide": 0.01, "deep": 3e-4}``. For other model hyper-parameters, see API reference of
:class:`~libreco.algorithms.WideDeep`.

.. code:: python3

    from libreco.algorithms import WideDeep

.. code:: python3

    model = WideDeep(
        task="ranking",
        data_info=data_info,
        embed_size=16,
        n_epochs=2,
        loss_type="cross_entropy",
        lr={"wide": 0.01, "deep": 3e-4},
        batch_size=2048,
        use_bn=True,
        hidden_units=(128, 64, 32),
    )
    
    model.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

::

    Epoch 1 elapsed: 3.053s
        train_loss: 0.6778
        eval log_loss: 0.5676
        eval roc_auc: 0.8005
        eval precision@10: 0.0277
        eval recall@10: 0.0409
        eval ndcg@10: 0.1119

    Epoch 2 elapsed: 3.008s
        train_loss: 0.4994
        eval log_loss: 0.4928
        eval roc_auc: 0.8373
        eval precision@10: 0.0321
        eval recall@10: 0.0506
        eval ndcg@10: 0.1384

We’ve trained the model for 2 epochs and evaluated the performance on
the eval data during training. Next we can evaluate on the *independent*
test data.

.. code:: python3

    >>> from libreco.evaluation import evaluate
    >>> evaluate(model=model, data=test_data, metrics=["loss", "roc_auc", "precision", "recall", "ndcg"])

.. parsed-literal::

    {'loss': 0.49392982253743395,
     'roc_auc': 0.8364561294428758,
     'precision': 0.03056640625,
     'recall': 0.05029253291880213,
     'ndcg': 0.12794099009836263}



Make Recommendation
-------------------

The recommend part is pretty straightforward. You can make
recommendation for one user or a batch of users.

.. code:: python3

    >>> model.recommend_user(user=1, n_rec=3)

.. parsed-literal::

    {1: array([ 260, 2858, 1210])}



.. code:: python3

    >>> model.recommend_user(user=[1, 2, 3], n_rec=3)

.. parsed-literal::

    {1: array([ 260, 2858, 1210]),
     2: array([527, 356, 480]),
     3: array([ 589, 2571, 1240])}



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

    >>> train_data = DatasetFeat.merge_trainset(train_data, loaded_data_info, merge_behavior=True)  # use loaded_data_info
    >>> eval_data = DatasetFeat.merge_evalset(eval_data, loaded_data_info)

    >>> train_data.build_negative_samples(loaded_data_info, seed=2022)  # use loaded_data_info
    >>> eval_data.build_negative_samples(loaded_data_info, seed=2222)


Then we construct a new model, and call :meth:`~libreco.algorithms.WideDeep.rebuild_model`
method to assign the old trained variables into the new model.

.. code:: python3

    >>> tf.compat.v1.reset_default_graph()  # need to reset graph in TensorFlow1

.. code:: python3

    new_model = WideDeep(
        task="ranking", 
        data_info=loaded_data_info,  # pass loaded_data_info
        embed_size=16, 
        n_epochs=2,
        loss_type="cross_entropy",
        lr={"wide": 0.01, "deep": 3e-4}, 
        batch_size=2048,
        use_bn=True,
        hidden_units=(128, 64, 32), 
    )
    
    new_model.rebuild_model(path="model_path", model_name="wide_deep", full_assign=True)

Finally, the training and recommendation parts are the same as before.

.. code:: python3

    new_model.fit(
        train_data, 
        verbose=2, 
        shuffle=True, 
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

::

    Epoch 1 elapsed: 2.955s
        train_loss: 0.4604
        eval log_loss: 0.4497
        eval roc_auc: 0.8678
        eval precision@10: 0.1015
        eval recall@10: 0.0715
        eval ndcg@10: 0.3106

    Epoch 2 elapsed: 2.657s
        train_loss: 0.4332
        eval log_loss: 0.4371
        eval roc_auc: 0.8760
        eval precision@10: 0.1043
        eval recall@10: 0.0740
        eval ndcg@10: 0.3189


.. code:: python3

    >>> new_model.recommend_user(user=1, n_rec=3)

.. parsed-literal::

    {1: array([2858, 1259, 3175])}

.. code:: python3

    >>> new_model.recommend_user(user=[1, 2, 3], n_rec=3)

.. parsed-literal::

    {1: array([2858, 1259, 3175]),
     2: array([1222, 1240,  858]),
     3: array([2858, 1580,  589])}

**This completes our tutorial!**

.. admonition:: Where to go from here
    :class: Note

    For more examples, see the `examples/ <https://github.com/massquantity/LibRecommender/tree/master/examples>`__ folder on GitHub.

    For more usages, please head to :doc:`user_guide/index`.

    For serving a trained model, please head to :doc:`serving_guide/python`.
