import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xenso
from xenso.events import _season_months


def monthly(values, start="2000-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return xr.DataArray(np.asarray(values, dtype=float), coords=[("time", dates)])


@pytest.fixture(scope="module")
def year_month():
    """Values encode the date as year * 100 + month."""
    dates = pd.date_range("2000-01-01", "2004-12-01", freq="MS")
    return xr.DataArray(dates.year * 100.0 + dates.month, coords=[("time", dates)])


@pytest.mark.parametrize(
    "season,expected",
    [
        (12, [12]),
        ("DEC", [12]),
        ("dec", [12]),
        ("NDJ", [11, 12, 1]),
        ("DJF", [12, 1, 2]),
        ("MAM", [3, 4, 5]),
    ],
)
def test_season_months(season, expected):
    assert _season_months(season) == expected


@pytest.mark.parametrize("season", [0, 13, "XYZ", "D", "JFMAMJJASONDJ"])
def test_season_months_invalid(season):
    with pytest.raises(ValueError):
        _season_months(season)


class TestSeasonalSeries:
    def test_single_month(self, year_month):
        result = xenso.seasonal_series(year_month, "DEC")
        np.testing.assert_array_equal(result.year, np.arange(2000, 2005))
        np.testing.assert_array_equal(result, np.arange(2000, 2005) * 100 + 12)

    def test_within_year(self, year_month):
        result = xenso.seasonal_series(year_month, "MAM")
        np.testing.assert_array_equal(result, np.arange(2000, 2005) * 100 + 4)

    def test_crossing_year_labelled_by_first_month(self, year_month):
        result = xenso.seasonal_series(year_month, "NDJ")
        # the incomplete NDJ 2004 (no Jan 2005) is dropped
        np.testing.assert_array_equal(result.year, np.arange(2000, 2004))
        years = np.arange(2000, 2004)
        expected = (years * 100 + 11 + years * 100 + 12 + (years + 1) * 100 + 1) / 3
        np.testing.assert_allclose(result, expected)

    def test_extra_dimensions(self, year_month):
        data = year_month.expand_dims(lon=[0, 1])
        result = xenso.seasonal_series(data, "DEC")
        assert set(result.dims) == {"year", "lon"}


class TestDetectEvents:
    @pytest.fixture(scope="class")
    def index(self):
        values = np.zeros(20 * 12)
        values[5 * 12 + 11] = 3  # Dec 2005
        values[10 * 12 + 11] = -3  # Dec 2010
        return monthly(values)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_nino_and_nina(self, index, normalize):
        nino = xenso.detect_events(index, normalize=normalize)
        nina = xenso.detect_events(index, kind="nina", normalize=normalize)
        np.testing.assert_array_equal(nino.year, [2005])
        np.testing.assert_array_equal(nina.year, [2010])
        np.testing.assert_allclose(nino, 3)

    def test_normalized_threshold(self, index):
        # std of the December values is sqrt(18 / 20) ~ 0.95, so 3.5 std ~ 3.3 > 3
        assert xenso.detect_events(index, threshold=3.5).size == 0
        assert xenso.detect_events(index, threshold=3.5, normalize=False).size == 0
        assert xenso.detect_events(index, threshold=3.0).size == 1

    def test_other_season(self, index):
        assert xenso.detect_events(index, season="MAM").size == 0

    def test_errors(self, index):
        with pytest.raises(ValueError):
            xenso.detect_events(index.expand_dims(lon=[0, 1]))
        with pytest.raises(ValueError):
            xenso.detect_events(index, kind="neutral")

    def test_observed_events(self, ersstv5):
        anom = xenso.compute_anomaly(
            xenso.region_mean(ersstv5, "nino34"), base_period=("1991-01-01", "2020-12-31")
        )
        index = xenso.smooth(xenso.detrend(anom))
        nino = xenso.detect_events(index).year.values
        nina = xenso.detect_events(index, kind="nina").year.values
        assert {1997, 2009, 2015}.issubset(nino)
        assert {1998, 2010, 2020}.issubset(nina)
        assert not set(nino) & set(nina)


class TestEventWindows:
    @pytest.fixture(scope="class")
    def months(self):
        """Values are the number of months since year 0, the index used internally."""
        dates = pd.date_range("2000-01-01", "2009-12-01", freq="MS")
        return xr.DataArray(dates.year * 12.0 + dates.month - 1, coords=[("time", dates)])

    def test_window(self, months):
        result = xenso.event_windows(months, [2004, 2005])
        assert result.dims == ("year", "lag")
        np.testing.assert_array_equal(result.lag, np.arange(-35, 37))
        np.testing.assert_array_equal(result.year, [2004, 2005])
        np.testing.assert_array_equal(result.sel(lag=0), [2004 * 12 + 11, 2005 * 12 + 11])
        # Jan of Y - 2 to Dec of Y + 3
        assert result.sel(year=2004).isel(lag=0) == 2002 * 12
        assert result.sel(year=2004).isel(lag=-1) == 2007 * 12 + 11

    def test_peak_month(self, months):
        result = xenso.event_windows(months, [2004], month=1)
        np.testing.assert_array_equal(result.lag, np.arange(-24, 48))
        assert result.sel(lag=0) == 2004 * 12

    def test_outside_record_is_nan(self, months):
        result = xenso.event_windows(months, [2000, 2009], window=4)
        assert result.sel(year=2000).isnull().sum() == 12
        assert result.sel(year=2009).isnull().sum() == 24
        assert result.sel(year=2000, lag=0) == 2000 * 12 + 11

    def test_detect_events_output(self, months):
        events = xr.DataArray([1.0, 2.0], coords=[("year", [2003, 2006])])
        result = xenso.event_windows(months, events)
        np.testing.assert_array_equal(result.year, [2003, 2006])

    def test_peak_only(self, months):
        result = xenso.event_windows(months.expand_dims(lon=[0, 1]), [2003, 2006], window=None)
        assert set(result.dims) == {"year", "lon"}
        np.testing.assert_array_equal(result.sel(lon=0), [2003 * 12 + 11, 2006 * 12 + 11])

    def test_composite(self, months):
        result = xenso.composite(months, [2003, 2005])
        np.testing.assert_array_equal(result.sel(lag=0), 2004 * 12 + 11)

    @pytest.mark.parametrize("window", [0, 3])
    def test_invalid_window(self, months, window):
        with pytest.raises(ValueError):
            xenso.event_windows(months, [2004], window=window)

    def test_not_monthly(self):
        daily = xr.DataArray(
            np.zeros(60), coords=[("time", pd.date_range("2000-01-01", periods=60, freq="D"))]
        )
        with pytest.raises(ValueError):
            xenso.event_windows(daily, [2000])


class TestEventDuration:
    @pytest.fixture(scope="class")
    def lifecycle(self):
        values = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        return xr.DataArray(values, coords=[("lag", np.arange(-3, 5))])

    def test_nino(self, lifecycle):
        # lags -2..0 before the peak, lags 1..2 after it
        result = xenso.event_duration(lifecycle, 0.5)
        assert result == 5
        assert result.attrs["units"] == "months"

    def test_nina(self, lifecycle):
        assert xenso.event_duration(-lifecycle, 0.5, kind="nina") == 5
        assert xenso.event_duration(lifecycle, 0.5, kind="nina") == 0

    def test_missing_values_stop_count(self, lifecycle):
        assert xenso.event_duration(lifecycle.where(lifecycle.lag != -1), 0.5) == 3

    def test_vectorized(self, lifecycle):
        stacked = xr.concat([lifecycle, lifecycle.where(lifecycle.lag != 1)], dim="year")
        np.testing.assert_array_equal(xenso.event_duration(stacked, 0.5), [5, 3])

    def test_invalid_kind(self, lifecycle):
        with pytest.raises(ValueError):
            xenso.event_duration(lifecycle, 0.5, kind="neutral")
