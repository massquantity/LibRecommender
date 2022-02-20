import functools
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from libreco.data import DatasetPure
from libreco.utils.similarities import cosine_sim, pearson_sim, jaccard_sim
from tests.utils_data import prepare_pure_data

raw_data = """
user,item,label
1,8,2
3,2,4
4,1,4
1,5,5
2,3,3
4,1,4
5,3,3
3,2,4
2,6,2
6,3,5
"""


@pytest.fixture
def pure_data():
    pd_data = pd.read_csv(StringIO(raw_data), header=0)
    return DatasetPure.build_trainset(pd_data)


# test small dataset
@pytest.mark.parametrize("func", [cosine_sim, pearson_sim, jaccard_sim])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("min_common", [1, 3])
def test_similarities(pure_data, func, num_threads, min_common):
    data, data_info = pure_data
    user_interaction = data.sparse_interaction
    item_interaction = user_interaction.T.tocsr()
    sim_func = functools.partial(
        func,
        sparse_data_x=item_interaction,
        sparse_data_y=user_interaction,
        num_x=data_info.n_items,
        num_y=data_info.n_users,
        block_size=None,
        num_threads=num_threads,
        min_common=min_common,
    )
    forward_sim = sim_func(mode="forward")
    invert_sim = sim_func(mode="invert")
    with pytest.raises(ValueError):
        _ = sim_func(mode="another")

    assert isinstance(forward_sim, csr_matrix)
    assert isinstance(invert_sim, csr_matrix)
    assert forward_sim.shape == invert_sim.shape
    np.testing.assert_array_equal(forward_sim.toarray(), invert_sim.toarray())


# test larger dataset
@pytest.mark.parametrize("func", [cosine_sim, pearson_sim, jaccard_sim])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("min_common", [1, 3])
def test_similarities_large(prepare_pure_data, func, num_threads, min_common):
    _, data, _, data_info = prepare_pure_data
    user_interaction = data.sparse_interaction
    item_interaction = user_interaction.T.tocsr()
    sim_func = functools.partial(
        func,
        sparse_data_x=item_interaction,
        sparse_data_y=user_interaction,
        num_x=data_info.n_items,
        num_y=data_info.n_users,
        block_size=None,
        num_threads=num_threads,
        min_common=min_common,
    )
    forward_sim = sim_func(mode="forward")
    invert_sim = sim_func(mode="invert")
    with pytest.raises(ValueError):
        _ = sim_func(mode="another")

    assert isinstance(forward_sim, csr_matrix)
    assert isinstance(invert_sim, csr_matrix)
    assert forward_sim.shape == invert_sim.shape
    np.testing.assert_array_equal(forward_sim.toarray(), invert_sim.toarray())
