In this section we describe some implementation details for [algorithms](https://github.com/massquantity/LibRecommender#references) in LibRecommender. In general, we try to follow the same settings used in the reference papers when implementing these algorithms. But in some cases we find it useful to change or extend these settings for better performance or speed, or it is necessary to adjust them in order to fit in with the whole process in LibRecommender. As we will explain in detail below.



## UserCF / ItemCF

The traditional implementation of UserCF / ItemCF is pre-allocating a user-user or item-item similarity matrix, then computing similarities between all users/items and fill in the matrix. However, this can be problematic for big data, because allocating a full user-user or item-item matrix may use a lot of memory. For example, for only about 100 thousand items, a (100,000, 100,000) matrix of numpy float64 will consume approximately 70 GB memory. So in LibRecommender we mainly use [scipy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html) to store similarity matrix and save memory.

Furthermore, computing all the user-user or item-item similarities requires iterating all data in for loops, which can be extremely slow in pure Python, so we use Cython with multi-threading to bypass the [Python GIL](https://wiki.python.org/moin/GlobalInterpreterLock). Also for the same purpose, we apply the [inverted index](https://en.wikipedia.org/wiki/Inverted_index) technic when computing similarities for better speed. The implementation is in [**_similarities.pyx**](https://github.com/massquantity/LibRecommender/blob/master/libreco/utils/_similarities.pyx), and users can choose whether to use forward index or inverted index by setting the `mode` parameter:

```python
>>> model = UserCF(..., mode="invert") # or "forward", note that "forward" mode can be much slower than "invert" mode
```

The concrete similarity metrics are [`cosine`](https://en.wikipedia.org/wiki/Cosine_similarity), [`pearson`](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and [`jaccard`](https://en.wikipedia.org/wiki/Jaccard_index). `pearson` is more suitable for rating task, and `jaccard` is more suitable for ranking task, whereas `cosine` is suitable for both. Users can choose them by setting `sim_type` parameter:

```python
>>> model = ItemCF(..., sim_type="cosine")
```



## FM / DeepFM

The FM and FM part of DeepFM is actually an implementation of [NFM](https://arxiv.org/pdf/1708.05027.pdf), which generalizes FM. The main difference is that in FM the final dimension of interaction layer is added element-wisely, whereas in NFM it is fed into DNN and gets final prediction. We found that NFM had a slightly better performance than FM.



## YouTubeRetrieval / YouTubeRanking

YouTubeRetrieval corresponds to the *candidate generation* stage of the [paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf), and YouTubeRanking corresponds to the *ranking* stage. For YouTubeRetrieval, The paper stated that negative sampling was used to alleviate extreme multi-class problem. So in our implementation [tf.nn.sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss) and  [tf.nn.nce_loss](https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss) are used, and you can choose one of them by specifying the `loss_type` parameter. However, there are two caveats when using these sampling techniques in TensorFlow:

By default, the `sampled_values` in [tf.nn.sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss) and  [tf.nn.nce_loss](https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss) uses [tf.random.log_uniform_candidate_sampler](https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler), which samples candidates based on log-uniform distribution. This basically means items with higher popularity will be more likely sampled. This is actually not very surprising because sampled-softmax and nce are originally came from NLP area, and in NLP this setting is common. However, this may not be suitable in recommender system scenario, especially in large-scale retrieval problem. So in LibRecommender you can set the `sampler` parameter to `uniform` to make the model use [tf.random.uniform_candidate_sampler](https://www.tensorflow.org/api_docs/python/tf/random/uniform_candidate_sampler), which samples items from uniform distribution. The default value in `sampler` is indeed `uniform`, and if you change it to other value, the default log-uniform in TensorFlow will be used.

Another caveat worth mentioning is the `num_sampled` parameter in [tf.nn.sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss) and  [tf.nn.nce_loss](https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss). The doc states that this parameter means "The number of classes to randomly sample **per batch**". Well, this is counter-intuitive, because in LibRecommender the `num_neg` parameter means "number of negative samples per positive sample". If you set `num_sampled` to 1, the model will sample only 1 negative sample for a batch data. So the parameter controls this setting in LibRecommender is `num_sampled_per_batch`, and the default value is `batch_size`, which means every positive sample in a batch will get 1 negative sample. But of course you can change to the value you want.

```python
>>> model = YouTubeRetrieval(
        task,
        data_info,
    	loss_type="sampled_softmax",  # or "nce"
        num_sampled_per_batch=None,  # "None" will use batch_size, or can be set to 1, 1000, 9999 ...
    	sampler="uniform",
    )
```



## RNN4Rec / GRU4Rec

The paper stated that they used GRU for modeling session-based data. But of course we can use LSTM too by setting the `rnn_type` parameter.

```python
>>> model= RNN4Rec(
        task,
        data_info,
        rnn_type="lstm",  # or "gru"
    )
```



## WaveNet

At first glance it looks weird to have [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) in LibRecommender, since it's a model used for generating raw audio. But if you look at the paper closely,  the way they model audio waveforms using CNN can also be applied to user behavior sequence. So we can generate user embedding based on this technique.



## NGCF / LightGCN

The [NGCF](https://arxiv.org/pdf/1905.08108.pdf) and [LightGCN](https://arxiv.org/pdf/2002.02126.pdf) paper used BPR (*Bayesian Personalized Ranking*) loss, but in LibRecommender one can also choose other losses by setting the `loss_type` parameter.

```python
>>> ngcf = NGCF(
        "ranking",
        data_info,
        loss_type="cross_entropy",  # or "focal", "bpr", "max_margin"
    )
>>> lightgcn = LightGCN(
        "ranking",
        data_info,
        loss_type="bpr",
    )
```



## PinSage

In LibRecommender, there are two versions of PinSage implementation: PyTorch and DGL version. Since some users may find it difficult to install DGL on Windows platform (see [issue](https://github.com/dmlc/dgl/issues/3067)), we provide an additional PyTorch version. In general the DGL version is much faster, but the PyTorch version can have more control over sampling process.

The [paper](https://arxiv.org/pdf/1806.01973.pdf) used max-margin loss on item-item inner product score. We extend this setting in our implementation. In recommender system scenario this is called "i2i", and the other form is "u2i", which is also commonly used and combines user features and item features to compute scores. The parameter for controlling this is `paradigm`. Max-margin loss belongs to pairwise loss, but we can also use other losses. In LibRecommender you can use `cross_entropy`, `focal`, `bpr`, `max_margin` by setting the `loss_type` parameter.

Another important extension in LibRecommender is that users can choose which features to use freely, instead of using domain-specific features described in the paper. So you can use PinSage just like other feat models:

```python
>>> sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
>>> dense_col = ["age"]
>>> user_col = ["sex", "age", "occupation"]
>>> item_col = ["genre1", "genre2", "genre3"]
>>> train_data, data_info = DatasetFeat.build_trainset(train, user_col, item_col, sparse_col, dense_col)

>>> from libreco.algorithms import PinSage, PinSageDGL
>>> model = PinSage(  # PyTorch version
        task,
        data_info,
        loss_type="cross_entropy",  # or "focal", "bpr", "max_margin"
        paradigm="u2i",  # or "i2i"
    )
>>> model = PinSageDGL(  # DGL version
        task,
        data_info,
        loss_type="max_margin",
        paradigm="i2i",
    )
>>> model.fit(train_data)
```



## GraphSage

GraphSage was not originally designed for recommender system problem, but we have adapted it to fit in with LibRecommender. Just like PinSage, GraphSage also has PyTorch and DGL version. The main difference between them is that the PyTorch version only implemented `mean` aggregator, whereas the DGL version can use `mean`, `gcn`, `pool`, `lstm`, thanks to the [SAGEConv](https://docs.dgl.ai/en/latest/generated/dgl.nn.pytorch.conv.SAGEConv.html) in DGL library.

```python
>>> sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
>>> dense_col = ["age"]
>>> user_col = ["sex", "age", "occupation"]
>>> item_col = ["genre1", "genre2", "genre3"]
>>> train_data, data_info = DatasetFeat.build_trainset(train, user_col, item_col, sparse_col, dense_col)

>>> from libreco.algorithms import GraphSage, GraphSageDGL
>>> model = GraphSage(  # PyTorch version
        task,
        data_info,
        loss_type="cross_entropy",  # or "focal", "bpr", "max_margin"
        paradigm="u2i",  # or "i2i"
    )
>>> model = GraphSageDGL(  # DGL version
        task,
        data_info,
        loss_type="focal",
        paradigm="i2i",
    	aggregator_type="mean",  # or "gcn", "pool", "lstm"
    )
>>> model.fit(train_data)
```

