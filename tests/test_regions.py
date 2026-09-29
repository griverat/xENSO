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


class TestRegionMean:
    @pytest.fixture(scope="class")
    def abs_lat(self):
        lat = np.arange(-89.5, 90)
        return xr.DataArray(
            np.tile(np.abs(lat)[:, None], (1, 360)),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": np.arange(0.5, 360)},
        )

    def test_nino_boxes_match(self):
        for key in ["12", "3", "34", "4"]:
            assert xenso.REGIONS[f"nino{key}"] == xenso.regions._REGIONS[key]

    def test_weighted(self, abs_lat):
        result = xenso.region_mean(abs_lat, "nino3_latext")
        lat = np.arange(-14.5, 15)
        np.testing.assert_allclose(result, np.average(np.abs(lat), weights=np.cos(np.deg2rad(lat))))
        np.testing.assert_allclose(xenso.region_mean(abs_lat, "nino3_latext", weighted=False), 7.5)

    def test_nino_regions_weighted_by_default(self, abs_lat):
        xr.testing.assert_allclose(xenso.nino_regions(abs_lat, "3"), xenso.region_mean(abs_lat, "nino3"))
        unweighted = xenso.nino_regions(abs_lat, "3", weighted=False)
        xr.testing.assert_allclose(unweighted, xenso.region_mean(abs_lat, "nino3", weighted=False))
        assert unweighted != xenso.nino_regions(abs_lat, "3")

    def test_skips_missing(self, abs_lat):
        masked = abs_lat.where(abs_lat.lon < 200)
        xr.testing.assert_allclose(
            xenso.region_mean(masked, "equatorial_pacific"),
            xenso.region_mean(abs_lat.sel(lon=slice(150, 200)), "equatorial_pacific"),
        )

    def test_unknown_region(self, abs_lat):
        with pytest.raises(ValueError):
            xenso.region_mean(abs_lat, "atlantic")
