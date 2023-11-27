import sys

import pytest

from libreco.data.consumed import _fill_empty, _merge_dedup, interaction_consumed


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Rust implementation only supports Python >= 3.7.",
)
def test_remove_consecutive_duplicates():
    user_indices = [1, 1, 1, 2, 2, 1, 2, 3, 2, 3]
    item_indices = [11, 11, 999, 0, 11, 11, 999, 11, 999, 0]
    user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
    assert isinstance(user_consumed, dict)
    assert isinstance(item_consumed, dict)
    assert isinstance(user_consumed[1], list)
    assert isinstance(item_consumed[11], list)
    assert user_consumed[1] == [11, 999, 11]
    assert user_consumed[2] == [0, 11, 999]
    assert user_consumed[3] == [11, 0]
    assert item_consumed[11] == [1, 2, 1, 3]
    assert item_consumed[999] == [1, 2]
    assert item_consumed[0] == [2, 3]


@pytest.mark.skipif(
    sys.version_info[:2] >= (3, 7),
    reason="Specific python 3.6 implementation",
)
def test_remove_duplicates():
    user_indices = [1, 1, 1, 2, 2, 1, 2, 3, 2, 3]
    item_indices = [11, 11, 999, 0, 11, 11, 999, 11, 999, 0]
    user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
    assert isinstance(user_consumed, dict)
    assert isinstance(item_consumed, dict)
    assert isinstance(user_consumed[1], list)
    assert isinstance(item_consumed[11], list)
    assert user_consumed[1] == [11, 999]
    assert user_consumed[2] == [0, 11, 999]
    assert user_consumed[3] == [11, 0]
    assert item_consumed[11] == [1, 2, 3]
    assert item_consumed[999] == [1, 2]
    assert item_consumed[0] == [2, 3]


def test_merge_remove_duplicates():
    num = 3
    old_consumed = {0: [1, 2, 3], 1: [4, 5]}
    new_consumed = {0: [2, 1], 2: [7, 8]}
    consumed = _merge_dedup(new_consumed, num, old_consumed)
    assert consumed[0] == [1, 2, 3, 2, 1]
    assert consumed[1] == [4, 5]
    assert consumed[2] == [7, 8]


def test_no_merge():
    num = 4
    old_consumed = {0: [1, 2, 3], 1: [4, 5], 2: [0], 3: [99]}
    new_consumed = {0: [2, 1], 2: [7, 8]}
    consumed = _fill_empty(new_consumed, num, old_consumed)
    assert consumed[0] == [2, 1]
    assert consumed[1] == [4, 5]
    assert consumed[2] == [7, 8]
    assert consumed[3] == [99]
