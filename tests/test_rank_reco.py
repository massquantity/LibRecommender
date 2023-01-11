import numpy as np
import pytest

from libreco.recommendation import rank_recommendations


def test_rank_reco():
    preds = np.array([-0.1, -0.01, 0, 0.1, 0.01])
    n_rec = 2
    n_items = 5
    consumed = [3, 4]
    id2item = {0: 10, 1: 11, 2: 22, 3: 33, 4: 44}

    with pytest.raises(ValueError):
        _ = rank_recommendations(
            "ranking",
            preds,
            n_rec + 10,  # n_rec exceeds n_items
            n_items,
            consumed,
            id2item,
            inner_id=True,
            filter_consumed=True,
            random_rec=False,
            return_scores=False,
        )

    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec,
        n_items,
        consumed,
        id2item,
        inner_id=True,
        filter_consumed=True,
        random_rec=False,
        return_scores=False,
    )
    assert len(rec_items) == n_rec
    np.testing.assert_array_equal(rec_items, [2, 1])

    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec + 2,  # can't filter consumed case
        n_items,
        consumed,
        id2item,
        inner_id=True,
        filter_consumed=True,
        random_rec=False,
        return_scores=False,
    )
    assert len(rec_items) == n_rec + 2
    np.testing.assert_array_equal(rec_items, [3, 4, 2, 1])

    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec,
        n_items,
        consumed,
        id2item,
        inner_id=False,  # return original item id
        filter_consumed=False,
        random_rec=False,
        return_scores=False,
    )
    assert len(rec_items) == n_rec
    np.testing.assert_array_equal(rec_items, [33, 44])

    _, scores = rank_recommendations(
        "ranking",
        preds,
        n_rec,
        n_items,
        consumed,
        id2item,
        inner_id=False,
        filter_consumed=True,
        random_rec=False,
        return_scores=True,
    )
    assert len(scores) == n_rec
    for i in range(1, len(scores)):
        assert scores[i - 1] >= scores[i]


def test_rank_random():
    preds = np.array([-0.1, -1e8, 0, 0.1, 0.01])  # small pred to zero prob case
    n_rec = 2
    n_items = 5
    consumed = [3, 4]
    id2item = {0: 10, 1: 11, 2: 22, 3: 33, 4: 44}

    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec,
        n_items,
        consumed,
        id2item,
        inner_id=True,
        filter_consumed=True,
        random_rec=True,
        return_scores=False,
    )
    assert len(rec_items) == n_rec
    assert 0 in rec_items
    assert 2 in rec_items

    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec + 2,
        n_items,
        consumed,
        id2item,
        inner_id=True,
        filter_consumed=True,
        random_rec=True,
        return_scores=False,
    )
    assert len(rec_items) == n_rec + 2
    assert 1 not in rec_items

    _, scores = rank_recommendations(
        "ranking",
        preds,
        n_rec,
        n_items,
        consumed,
        id2item,
        inner_id=False,
        filter_consumed=True,
        random_rec=True,
        return_scores=True,
    )
    assert len(scores) == n_rec
    for i in range(1, len(scores)):
        assert scores[i - 1] >= scores[i]

    preds = np.array([1e8, -0.01, 1e7, 0.1, 0.01])  # big pred to zero prob case
    n_rec = 3
    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec,
        n_items,
        consumed,
        id2item,
        inner_id=True,
        filter_consumed=True,
        random_rec=True,
        return_scores=False,
    )
    assert len(rec_items) == n_rec
    np.testing.assert_array_equal(np.sort(rec_items), [0, 1, 2])

    rec_items = rank_recommendations(
        "ranking",
        preds,
        n_rec + 2,
        n_items,
        consumed,
        id2item,
        inner_id=True,
        filter_consumed=True,
        random_rec=True,
        return_scores=False,
    )
    assert len(rec_items) == n_rec + 2
    np.testing.assert_array_equal(np.sort(rec_items), [0, 1, 2, 3, 4])
