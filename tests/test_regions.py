import numpy as np
import pytest
import xarray as xr

import xenso


class TestEnregions:
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
