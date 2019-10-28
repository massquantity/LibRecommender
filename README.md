# LibRecommender

## Overview

**LibRecommender** is an easy-to-use recommender system focused on end-to-end recommendation. The main features are:

+ Implement a number of popular recommendation algorithms such as SVD, DeepFM, BPR etc.

+ A hybrid system, allow user to use either collaborative-filtering or content-based features.

+ Ease of memory usage, automatically convert categorical features to sparse representation.

+ Suitable for both explicit and implicit datasets, and negative sampling can be used for implicit dataset.

+ Making use of Cython or Tensorflow to accelerate model training.

+ Provide end-to-end workflow, i.e. data handling / preprocessing -> model training -> evaluate -> serving.



## Usage

##### _pure collaborative-filtering example_ : 

```python
from libreco.dataset import DatasetPure   # pure data, algorithm svd++
from libreco.algorithms import SVDpp

conf = {
    "data_path": "path/to/your/data",
    "length": "all",
}

dataset = DatasetPure()
dataset.build_dataset(**conf)

svd = SVDpp(n_factors=32, n_epochs=200, lr=0.001, batch_size=4096, task="rating")
svd.fit(dataset, verbose=1)
print(svd.predict(1, 2))	     # predict preference of user 1 to item 2
print(svd.recommend_user(1, 7))	 # recommend 7 items for user 1
```

##### _include features example_ : 

```python
from libreco.dataset import DatasetFeat   # feat data, algorithm DeepFM
from libreco.algorithms import DeepFmFeat

conf = {
    "data_path": "path/to/your/data",
    "length": 500000,
    "user_col": 0,
    "item_col": 1,
    "label_col": 2,
    "numerical_col": [4],
    "categorical_col": [3, 5, 6, 7, 8],
    "merged_categorical_col": None,
    "user_feature_cols": [3, 4, 5],
    "item_feature_cols": [6, 7, 8],
    "convert_implicit": True,
    "build_negative": True,
    "num_neg": 2,
    "sep": ",",
}

dataset = DatasetFeat(include_features=True)
dataset.build_dataset(**conf)

dfm = DeepFmFeat(lr=0.0002, n_epochs=10000, reg=0.1, embed_size=50,
                 batch_size=2048, dropout_rate=0.0, task="ranking", neg_sampling=True)
dfm.fit(dataset, pre_sampling=False, verbose=1)
print(dfm.predict(1, 10))             # predict preference of user 1 to item 10
print(dfm.recommend_user(1, 7))   # recommend 7 items for user 1
```


## Data Format
JUST normal data format, each line represents a sample. By default, model assumes that `user`, `item`, and `label` column index are 0, 1, and 2, respectively. But you need to specify `user`, `item`, and `label` column index if that’s not the case. For Example, the `movielens-1m` dataset:

> 1::1193::5::978300760<br>
> 1::661::3::978302109<br>
> 1::914::3::978301968<br>
> 1::3408::4::978300275

leads to the following settings in `conf` dict : `"user_col": 0,  "item_col": 1,  "label_col": 2, "sep": "::"` .

Besides, if you want to use some other meta features (e.g., age, sex, category etc.), `numerical` and `categorical` column index must be assigned. For example, `"numerical_col": [4], "categorical_col": [3, 5, 6, 7, 8]`, which means all features must be in a same table.



## Installation & Dependencies 

From pypi:  `pip install LibRecommender`



- Python 3.5 +
- tensorflow >= 1.12
- numpy >= 1.15.4
- pandas >= 0.23.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.0





## References

|     Algorithm     | Category | Paper                                                        |
| :---------------: | :------: | :----------------------------------------------------------- |
| userKNN / itemKNN |   pure   | [Item-Based Collaborative Filtering Recommendation Algorithms](http://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf) |
|        SVD        |   pure   | [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) |
|      SVD ++       |   pure   | [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](https://dl.acm.org/citation.cfm?id=1401944) |
|     superSVD      |   pure   | [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](https://dl.acm.org/citation.cfm?id=1401944) |
|        ALS        |   pure   | 1. [Matrix Completion via Alternating Least Square(ALS)](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf) / <br>2. [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) / <br>3. [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf) |
|        NCF        |   pure   | [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf) |
|        BPR        |   pure   | [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) |
|    Wide & Deep    |   feat   | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) |
|        FM         |   feat   | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
|      DeepFM       |   feat   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf) |
|    YouTubeRec     |   feat   | [Deep Neural Networks for YouTube Recommendations](<https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf>) |


## License

#### MIT

<br>