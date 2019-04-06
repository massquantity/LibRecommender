import time
from recommender_project.dataset.Dataset import Dataset
from recommender_project.algorithms import user_KNN, item_KNN
from recommender_project.evaluate import rmse_user_knn
from recommender_project.utils.baseline_estimates import baseline_als, baseline_sgd


if __name__ == "__main__":
    t0 = time.time()
#    loaded_data = Dataset.load_dataset(data_path="ml-1m/ratings.dat")
    dataset = Dataset()
    dataset.build_dataset(data_path="ml-1m/ratings.dat", length=100000, shuffle=True)
    train, test = dataset.train_user, dataset.test_user_indices
#    print(train[1428])
#    print(dataset.train_item[2428])
#    print(train[1428][2428])
    user_knn = user_KNN.userKNN(sim_option="cosine", k=50, min_support=1, baseline=False)
    user_knn.fit(dataset)
#    print(user_knn.predict(1428, 2428))
#    print(rmse_user_knn(user_knn, dataset, mode="train"))
#    print(rmse_user_knn(user_knn, dataset, mode="test"))
    print(user_knn.topN(1, 10, 5, random_rec=False))
    print(user_knn.topN(1, 10, 5, random_rec=True))
    print("train + test time: {:.4f}".format(time.time() - t0))
