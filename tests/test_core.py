import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.variable import MissingDimensionsError

import xenso


def test_compute_climatology():
    dates = pd.date_range("1981-01-01", "2010-12-31", freq="ME")
    data = xr.DataArray(np.tile(np.arange(12), 30), coords=[("time", dates)])

    result = xenso.compute_climatology(data)
    expected = xr.DataArray(np.arange(12), coords=[("month", np.arange(1, 13))])
    xr.testing.assert_equal(result, expected)

    result = xenso.compute_climatology(data, ("2000-01-01", "2006-12-31"))
    xr.testing.assert_equal(result, expected)


def test_compute_anomaly():
    dates = pd.date_range("1981-01-01", "2010-12-31", freq="ME")
    data = xr.DataArray(np.tile(np.arange(12), 30), coords=[("time", dates)])
    climatology = xr.DataArray(np.arange(12), coords=[("month", np.arange(1, 13))])

    result = xenso.compute_anomaly(data, climatology=climatology)
    expected = xr.full_like(data, 0)
    expected["month"] = ("time", np.tile(np.arange(1, 13), 30))
    xr.testing.assert_equal(result, expected)

    result = xenso.compute_anomaly(data, base_period=("2000-01-01", "2005-12-31"))
    xr.testing.assert_equal(result, expected)

    # test error
    with pytest.raises(ValueError):
        xenso.compute_anomaly(data)

    with pytest.raises(MissingDimensionsError):
        xenso.compute_anomaly(data.rename({"time": "month"}), climatology=climatology)
