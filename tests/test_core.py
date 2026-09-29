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


class TestSmooth:
    @pytest.fixture(scope="class")
    def spike(self):
        values = np.zeros(11)
        values[5] = 9.0
        dates = pd.date_range("2000-01-01", periods=11, freq="MS")
        return xr.DataArray(values, coords=[("time", dates)])

    def test_triangle_weights(self, spike):
        result = xenso.smooth(spike)
        # 1-2-3-2-1 weights, edges dropped
        np.testing.assert_allclose(result, [0, 1, 2, 3, 2, 1, 0])
        xr.testing.assert_equal(result.time, spike.time[2:-2])

    @pytest.mark.parametrize("method", ["triangle", "square", "gaussian"])
    def test_preserves_linear_trend(self, method):
        data = xr.DataArray(np.arange(12.0), dims="time")
        np.testing.assert_allclose(xenso.smooth(data, 5, method=method), np.arange(2.0, 10.0))

    def test_square(self, spike):
        np.testing.assert_allclose(xenso.smooth(spike, 3, method="square"), [0, 0, 0, 3, 3, 3, 0, 0, 0])

    def test_missing_values(self):
        data = xr.DataArray([1.0, np.nan, np.nan, np.nan, 4.0, 5.0, 6.0], dims="time")
        result = xenso.smooth(data)
        # more than half of the window missing -> NaN, otherwise weights renormalized
        assert result[:2].isnull().all()
        np.testing.assert_allclose(result[2], (3 * 4 + 2 * 5 + 1 * 6) / 6)

    def test_other_dimension(self, spike):
        data = spike.expand_dims(lon=[0, 1]).transpose("time", "lon")
        result = xenso.smooth(data)
        assert result.dims == ("time", "lon")
        np.testing.assert_allclose(result.sel(lon=1), [0, 1, 2, 3, 2, 1, 0])

    @pytest.mark.parametrize("kwargs", [{"window": 4}, {"window": 0}, {"method": "hann"}])
    def test_errors(self, spike, kwargs):
        with pytest.raises(ValueError):
            xenso.smooth(spike, **kwargs)


class TestDetrend:
    @pytest.fixture(scope="class")
    def trend(self):
        dates = pd.date_range("2000-01-01", periods=48, freq="MS")
        return xr.DataArray(2.0 + 0.5 * np.arange(48), coords=[("time", dates)])

    def test_keep_mean(self, trend):
        np.testing.assert_allclose(xenso.detrend(trend), trend.mean())

    def test_zero_mean(self, trend):
        np.testing.assert_allclose(xenso.detrend(trend, keep_mean=False), 0, atol=1e-8)

    def test_removes_trend_and_keeps_gaps(self, trend):
        signal = xr.DataArray(np.sin(np.arange(48) * 2 * np.pi / 12), coords=trend.coords)
        signal = signal.where(trend.time.dt.month != 3)
        result = xenso.detrend(trend + signal, keep_mean=False)
        assert result.isnull().sum() == 4
        xr.testing.assert_allclose(result, xenso.detrend(signal, keep_mean=False))
