import numpy as np
import pytest
import xarray as xr

import xenso


class TestNinoRegions:
    @pytest.fixture(scope="class")
    def dummy(self):
        return xr.DataArray(
            np.tile(np.arange(360), (180, 1)),
            dims=["lat", "lon"],
            coords={"lat": np.arange(-90, 90), "lon": np.arange(360)},
        )

    @pytest.mark.parametrize(
        "region,expected",
        [("12", 275), ("3", 240), ("34", 215), ("4", 185)],
    )
    def test_compute_nino_regions(self, dummy, region, expected):
        actual = xenso.nino_regions(dummy, region=region)
        np.testing.assert_allclose(actual, expected)

    def test_default_region_is_34(self, dummy):
        assert xenso.nino_regions(dummy) == xenso.nino_regions(dummy, region="34")

    def test_descending_lat(self, dummy):
        # nino_regions delegates normalization to preprocessing.normalize_coords
        flipped = dummy.sortby("lat", ascending=False)
        result = xenso.nino_regions(flipped, region="34")
        expected = xenso.nino_regions(dummy, region="34")
        xr.testing.assert_equal(result, expected)


class TestONI:
    BASE_PERIOD = ("1991-01-01", "2020-12-31")

    @pytest.fixture(scope="class")
    def oni(self, ersstv5):
        return xenso.oni(ersstv5, base_period=self.BASE_PERIOD)

    def test_returns_dataarray(self, oni):
        assert isinstance(oni, xr.DataArray)

    def test_time_dimension(self, oni, ersstv5):
        # centered 3-month rolling + dropna removes first and last time step
        assert len(oni.time) == len(ersstv5.time) - 2

    def test_is_3month_running_mean(self, oni, ersstv5):
        nino34_anom = xenso.compute_anomaly(
            xenso.nino_regions(ersstv5, region="34"),
            base_period=self.BASE_PERIOD,
        )
        expected = nino34_anom.rolling(time=3, center=True).mean().dropna("time")
        xr.testing.assert_allclose(oni, expected, rtol=1e-5)


class TestRONI:
    BASE_PERIOD = ("1991-01-01", "2020-12-31")

    @pytest.fixture(scope="class")
    def roni(self, ersstv5):
        return xenso.roni(ersstv5, base_period=self.BASE_PERIOD)

    @pytest.fixture(scope="class")
    def oni(self, ersstv5):
        return xenso.oni(ersstv5, base_period=self.BASE_PERIOD)

    def test_returns_dataarray(self, roni):
        assert isinstance(roni, xr.DataArray)

    def test_same_time_length_as_oni(self, roni, oni):
        assert len(roni.time) == len(oni.time)

    def test_monthly_variance_matches_oni(self, roni, oni):
        # the variance rescaling ensures rONI has the same monthly std as ONI
        roni_std = roni.groupby("time.month").std("time")
        oni_std = oni.groupby("time.month").std("time")
        xr.testing.assert_allclose(roni_std, oni_std, rtol=1e-1)
