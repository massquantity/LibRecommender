import time

import pytest

from libreco.utils.misc import colorize, time_block, time_func


@time_func
def long_work():
    time.sleep(0.1)
    print(colorize("done!", color="red", bold=True, highlight=True))


def test_misc():
    long_work()
    with time_block("long work2", verbose=0):
        time.sleep(0.1)
    with pytest.raises(RuntimeError):
        with time_block("long work2", verbose=0):
            raise RuntimeError
