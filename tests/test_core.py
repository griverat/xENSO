import numpy as np
import pandas as pd
import xarray as xr

import xenso


def test_compute_climatology():
    dates = pd.date_range("1981-01-01", "2010-12-31", freq="M")
    data = xr.DataArray(np.tile(np.arange(12), 30), coords=[("time", dates)])

    result = xenso.compute_climatology(data)
    expected = xr.DataArray(np.arange(12), coords=[("month", np.arange(1, 13))])
    xr.testing.assert_equal(result, expected)

    result = xenso.compute_climatology(data, ("2000-01-01", "2006-12-31"))
    xr.testing.assert_equal(result, expected)
