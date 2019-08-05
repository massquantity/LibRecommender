import os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from pathlib import Path, PurePath
from libreco.dataset import DatasetPure, DatasetFeat
from libreco.algorithms import userKNN, FmFeat, WideDeep, WideDeepEstimator, WideDeep
from libreco import baseline_als
from libreco import NegativeSampling
from libreco.utils import export_model_pickle, export_model_joblib, export_model_tf, export_feature_transform
# from libreco.utils import export_model_pickle, export_model_joblib


if __name__ == "__main__":
    t0 = time.time()
#    dataset = DatasetPure()
#    path = str(Path.joinpath(Path(__file__).parent, "ml-1m", "ratings.dat"))
#    dataset.build_dataset(data_path=path, sep="::", length=10000, shuffle=True,
#                          convert_implicit=False, build_negative=False, num_neg=10, batch_size=256)
#    dataset.leave_k_out_split(4, data_path="ml-1m/ratings.dat", length=100000, sep="::",
#                              convert_implicit=True, build_negative=True, batch_size=256, num_neg=1)
#   print("data size: ", len(dataset.train_user_implicit) + len(dataset.test_user_implicit))
#    print("data processing time: {:.2f}".format(time.time() - t0))
    '''
    conf = {
        "data_path": "ml-1m/merged_data.csv",
        "sep": ",",
        "header": None,
        "col_names": ['user', 'item', 'label', 'sex', 'age', 'occupation', 'title', 'genre1', 'genre2', 'genre3'],
        "length": 100000,
        "user_col": 'user',
        "item_col": 'item',
        "label_col": 'label',
        "user_feature_cols": ["sex", "age", "occupation"],
        "item_feature_cols": ['title', 'genre1', 'genre2', 'genre3'],
        "convert_implicit": True,
        "build_negative": True,
        "num_neg": 2,
    }

    dataset = DatasetFeat(include_features=True)
    dataset.load_pandas(**conf)
    print("data processing time: {:.2f}".format(time.time() - t0))
#    import pstats, cProfile
#    cProfile.run("dataset.load_pandas(**conf)")
#    s = pstats.Stats("Profile.prof")
#    s.strip_dirs().sort_stats("time").print_stats()

    '''

    conf = {
        "data_path": "ml-1m/merged_data.csv",
        "length": "all",
        "user_col": 0,
        "item_col": 1,
        "label_col": 2,
        "numerical_col": None,
        "categorical_col": [3, 4, 5, 6],
        "merged_categorical_col": [[7, 8, 9]],
        "item_feature_cols": [6, 7, 8, 9],
        "convert_implicit": False,
        "build_negative": False,
        "num_neg": 2,
        "batch_size": 256,
        "sep": ",",
    }

    dataset = DatasetFeat(include_features=True)
    dataset.build_dataset(**conf)
#    dataset.build_dataset(data_path="ml-1m/merged_data.csv", length="all", user_col=0, item_col=1, label_col=2,
#                          numerical_col=None, categorical_col=[3, 4, 5, 6], merged_categorical_col=[[7, 8, 9]],
#                          item_sample_col=[6, 7, 8, 9],
#                          convert_implicit=True, build_negative=True, num_neg=1, batch_size=256)
#                         numerical_col=None, categorical_col=[3, 4, 5, 6, 7, 8], merged_categorical_col=None)
#    dataset.leave_k_out_split(4, data_path="ml-1m/merged_data.csv", length="all", sep=",", shuffle=True,
#                              user_col=0, item_col=1, label_col=2, numerical_col=None, categorical_col=[3, 4, 5],
#                              merged_categorical_col=[[6, 7, 8]])
#    print("data size: ", len(dataset.train_indices_implicit) + len(dataset.test_indices_implicit))
    print("data processing time: {:.2f}".format(time.time() - t0))
    print()

#    user_knn = userKNN(sim_option="msd", k=40, min_support=0, baseline=False)
#    user_knn.fit(dataset)
#    t1 = time.time()
#    print("predict: ", user_knn.predict(0, 5))
#    print("predict time: ", time.time() - t1)
#    current_path = Path(".").resolve()
#    joblib_path = str(Path.joinpath(current_path, "serving/models/user_knn.jb"))
#    export_model_joblib(joblib_path, user_knn)

#    t2 = time.time()
#    print("rmse: ", user_knn.evaluate(dataset, 1000000))
#    print("evaluate time: ", time.time() - t2)

#    t4 = time.time()
#    print("rmse train:", rmse_knn(user_knn, dataset, mode="train"))
#    print("rmse train time: ", time.time() - t4)
#    print("rmse test: ", rmse_knn(user_knn, dataset, mode="test"))

#    svd = SVD.SVD(n_factors=10, n_epochs=200, reg=10.0)
#    svd.fit(dataset)
#    print(rmse_svd(svd, dataset, mode="train"))
#    print(rmse_svd(svd, dataset, mode="test"))
#    print(svd.topN(1, 5, random_rec=False))
#    print(svd.topN(1, 5, random_rec=True))

#    svd = SVD.SVD_tf(n_factors=100, n_epochs=10, lr=0.01, reg=0.1,
#                     batch_size=1280, batch_training=True)
#    svd.fit(dataset, data_mode="structure")
#    print(svd.predict(1,2))
#    print(rmse_svd(svd, dataset, mode="train"))
#    print(rmse_svd(svd, dataset, mode="test"))

#    svd = SVD.SVDBaseline(n_factors=30, n_epochs=20000, lr=0.001, reg=0.1,
#                          batch_size=256, batch_training=True)
#    svd.fit(dataset)
#    print(rmse_svd(svd, dataset, mode="train"))
#    print(rmse_svd(svd, dataset, mode="test"))

#    svdpp = SVDpp.SVDpp(n_factors=30, n_epochs=20000, lr=0.001, reg=0.1,
#                        batch_size=256, batch_training=True)
#    svdpp.fit(dataset)
#    print(rmse_svd(svdpp, dataset, mode="train"))
#    print(rmse_svd(svdpp, dataset, mode="test"))

#    svdpp = SVDpp.SVDpp_tf(n_factors=100, n_epochs=10, lr=0.001, reg=0.1,
#                            batch_size=256, batch_training=True)  # 0.8579
#    svdpp.fit(dataset)
#    print(svdpp.predict(1,2))
#    print(rmse_svd(svdpp, dataset, mode="train"))
#    print(rmse_svd(svdpp, dataset, mode="test"))

#    superSVD = superSVD.superSVD(n_factors=200, n_epochs=10000, lr=0.001, reg=0.001,
#                                 batch_training=True, sim_option="cosine",
#                                 k=40, min_support=5)  # lr1, lr2 reg1, reg2
#    superSVD.fit(dataset)
#    print(superSVD.predict(1,2))
#    print(rmse_svd(superSVD, dataset, mode="train"))
#    print(rmse_svd(superSVD, dataset, mode="test"))

#    ncf = NCF.NCF(embed_size=32, lr=0.0007, batch_size=256, n_epochs=500)
#    ncf = NCF.NCF(embed_size=16, lr=0.001, batch_size=256, n_epochs=500)
#    ncf.fit(dataset)
#    print(ncf.predict(1,2))
#    print(AP_at_k(ncf, dataset, 1, 10))
#    print(MAP_at_k(ncf, dataset, 10))
#    print(rmse_tf(ncf, dataset, mode="train"))
#    print(rmse_tf(ncf, dataset, mode="test"))

#    wd = wide_deep.WideDeep(embed_size=16, n_epochs=1, batch_size=256, task="ranking")
#    wd.fit(dataset)
#    print(wd.predict(1, 2, "2001-1-8"))
#    print(wd.predict_user(1))

#    wdc = WideDeepEstimator(lr=0.01, embed_size=16, n_epochs=100, batch_size=256, task="ranking", cross_features=False)
    wdc = WideDeep(lr=0.01, embed_size=16, n_epochs=100, batch_size=256, dropout_rate=0.0, task="rating")
    wdc.fit(dataset)
    print(wdc.predict(1, 2))
#    t6 = time.time()
#    print(wdc.recommend_user(1, n_rec=10))
#    print("rec time: ", time.time() - t6)

    # reg=0.001, n_factors=32 reg=0.0001   0.8586  0.8515  0.8511
    # reg=0.0003, n_factors=64, 0.8488    0.8471 0.8453
#    fm = FM.FmPure(lr=0.0001, n_epochs=20000, reg=0.0, n_factors=16, batch_size=256, task="ranking")
#    fm = FmFeat(lr=0.0001, n_epochs=2, reg=0.0, n_factors=16, batch_size=256, task="ranking")
#    fm.fit(dataset, pre_sampling=True)
#    export_model_tf(fm, "FM", "1", simple_save=False)
#    current_path = Path(".").resolve()
#    fb_path = str(Path.joinpath(current_path, "serving/models/others/feature_builder.jb"))
#    conf_path = str(Path.joinpath(current_path, "serving/models/others/conf.jb"))
#    export_feature_transform(fb_path, conf_path, dataset.fb, conf)

#    num = {}
#    cat = {3: 'F', 4: 1, 5: 10, 6: 2452.0}
#    merge = {7: ["Drama", "missing", "missing"]}
#    indices, values = fm.dataset.fb.transform(cat, num, merge, 1, np.array([1]), np.array([1193]))
#    print(indices)
#    print(values)
#    fm.export_model(version="1", simple_save=False)
#    print(fm.predict(1, 2))

#    dfm = DeepFM.DeepFM(lr=0.0001, n_epochs=20000, reg=0.0, embed_size=8,
#                        batch_size=1024, dropout=0.0, task="ranking")
#    dfm.fit(dataset)
#    print(dfm.predict(1, 2))
#    print(dfm.predict_user(1))

#    iteration = len(dataset.train_user_indices) * 10000
#    bpr = BPR.BPR(lr=0.01, iteration=iteration)  # reg
#    bpr.fit(dataset, sampling_mode="bootstrap")
#    print(bpr.predict(1, 2))

#    bpr = BPR.BPR(lr=0.03, n_epochs=2, reg=0.0, n_factors=16, batch_size=64)
#    bpr.fit(dataset, sampling_mode="sgd")
#    print(bpr.predict(1, 2))

#    bpr = BPR.BPR_tf(lr=0.001, n_epochs=20, reg=0.0, n_factors=16, batch_size=64)
#    bpr.fit(dataset)
#    print(bpr.predict(1, 2))

#    bpr = BPR.BPR(lr=0.01, n_epochs=2000, reg=0.0, k=100)
#    bpr.fit(dataset, method="knn")
#    print(bpr.predict(1, 2, method="knn"))

    print("train + test time: {:.4f}".format(time.time() - t0))



