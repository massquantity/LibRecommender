import itertools

import numpy as np
import pytest

from libreco.utils.initializers import (
    he_init,
    truncated_normal,
    variance_scaling,
    xavier_init,
)


def test_initializers():
    np_rng = np.random.default_rng(42)
    mean, std, fan_in, fan_out, scale = 0.1, 0.01, 4, 2, 2.5
    variables = truncated_normal(np_rng, [3, 2], mean=0.1, scale=0.01)
    assert variables.shape == (3, 2)
    variables_in_range(variables, mean, std)

    variables = xavier_init(np_rng, fan_in, fan_out)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    variables_in_range(variables, mean, std)

    variables = he_init(np_rng, fan_in, fan_out)
    std = 2.0 / np.sqrt(fan_in + fan_out)
    variables_in_range(variables, mean, std)

    variables = variance_scaling(np_rng, scale, fan_in, fan_out, mode="fan_in")
    std = np.sqrt(scale / fan_in)
    variables_in_range(variables, mean, std)

    variables = variance_scaling(np_rng, scale, fan_in, fan_out, mode="fan_out")
    std = np.sqrt(scale / fan_out)
    variables_in_range(variables, mean, std)

    variables = variance_scaling(np_rng, scale, fan_in, fan_out, mode="fan_average")
    std = np.sqrt(2.0 * scale / (fan_in + fan_out))
    variables_in_range(variables, mean, std)

    with pytest.raises(ValueError):
        _ = variance_scaling(np_rng, scale, fan_in, fan_out, mode="unknown")


def variables_in_range(variables, mean, std):
    for v in itertools.chain.from_iterable(variables):
        assert (mean - 3 * std) < v < (mean + 3 * std)
