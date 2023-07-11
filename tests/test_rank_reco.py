import numpy as np
import pytest

from libreco.recommendation import rank_recommendations


def test_rank_reco():
    user_ids = [1, 2]
    preds = np.array([-0.1, -0.01, 0, 0.1, 0.01, 1, -2, 4, 5, 6])
    n_rec = 2
    n_items = 5
    consumed = {1: [3, 4], 2: [4]}

    with pytest.raises(ValueError):
        _ = rank_recommendations(
            "ranking",
            user_ids,
            preds,
            n_rec + 10,  # n_rec exceeds n_items
            n_items,
            consumed,
            filter_consumed=True,
            random_rec=False,
            return_scores=False,
        )

    rec_items = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec,
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=False,
        return_scores=False,
    )
    assert rec_items.shape == (2, 2)
    np.testing.assert_array_equal(rec_items[0], [2, 1])
    np.testing.assert_array_equal(rec_items[1], [3, 2])

    rec_items = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec + 2,  # can't filter consumed case
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=False,
        return_scores=False,
    )
    assert rec_items.shape == (2, 4)
    np.testing.assert_array_equal(rec_items[0], [3, 4, 2, 1])
    np.testing.assert_array_equal(rec_items[1], [3, 2, 0, 1])

    _, scores = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec,
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=False,
        return_scores=True,
    )
    assert scores.shape == (2, 2)
    for score in scores.tolist():
        for i in range(1, len(score)):
            assert score[i - 1] >= score[i]

    preds = np.array([[-0.1, -0.01, 0, 0.1, 0.01], [1, -2, 4, 5, 6]])  # 2d array
    rec_items = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec,
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=False,
        return_scores=False,
    )
    assert rec_items.shape == (2, 2)
    np.testing.assert_array_equal(rec_items[0], [2, 1])
    np.testing.assert_array_equal(rec_items[1], [3, 2])


def test_rank_random():
    user_ids = [1, 2]
    # fmt: off
    preds = np.array([-0.1, -1e8, 0, 0.1, 0.01, 1e8, -0.01, 1e7, 0.1, 0.01])  # inf probs
    n_rec = 2
    n_items = 5
    consumed = {1: [3, 4], 2: [4]}

    rec_items = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec,
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=True,
        return_scores=False,
    )
    assert rec_items.shape == (2, 2)
    assert 0 in rec_items[0]
    assert 2 in rec_items[0]
    assert 0 in rec_items[1]

    rec_items = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec + 2,
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=True,
        return_scores=False,
    )
    assert rec_items.shape == (2, 4)
    assert 1 not in rec_items[0]
    assert 1 in rec_items[1]

    _, scores = rank_recommendations(
        "ranking",
        user_ids,
        preds,
        n_rec,
        n_items,
        consumed,
        filter_consumed=True,
        random_rec=True,
        return_scores=True,
    )
    assert scores.shape == (2, 2)
    for score in scores.tolist():
        for i in range(1, len(score)):
            assert score[i - 1] >= score[i]
