import numpy as np
import pytest

import xenso


@pytest.mark.web
def test_meaning_of_life_url():
    ret = xenso.meaning_of_life_url()

    assert isinstance(ret, str)
    assert "Monty Python" in ret
    assert "Meaning of Life" in ret


def test_meaning_of_life():
    n = 2
    ret = xenso.meaning_of_life(n)
    assert isinstance(ret, np.ndarray)
    assert np.unique(ret) == 42
