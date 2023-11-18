import numpy as np

from libreco.prediction import predict_data_with_feats


def ptest_preds(model, task, pd_data, with_feats):
    user = pd_data.user.iloc[0]
    item = pd_data.item.iloc[0]
    pred = model.predict(user=user, item=item)
    # prediction in range
    if task == "rating":
        assert 1 <= pred <= 5
    else:
        assert 0 <= pred <= 1

    popular_pred = model.predict(
        user="cold user2", item="cold item2", cold_start="popular"
    )
    assert np.allclose(popular_pred, model.default_pred)

    cold_pred1 = model.predict(user="cold user1", item="cold item2")
    cold_pred2 = model.predict(user="cold user2", item="cold item2")
    assert cold_pred1 == cold_pred2

    if with_feats:
        assert len(predict_data_with_feats(model, pd_data[:5])) == 5
        model.predict(user=user, item=item, feats={"sex": "male", "genre_1": "crime"})
