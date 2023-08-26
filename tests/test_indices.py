import numpy as np
import pytest
import xarray as xr

import xenso


class TestECindex:
    @pytest.fixture(scope="class")
    def ec(self, ersstv5):
        return xenso.indices.ECindex(ersstv5, base_period=("1991", "2020"))

    def test_compute_ecindex(self, ec):
        expected = xr.open_dataset("tests/data/ecindex.nc")
        xr.testing.assert_allclose(ec.ecindex, expected, rtol=1e-3)

    def test_compute_ecindex_smooth(self, ec):
        expected = xr.open_dataset("tests/data/ecindex_smooth.nc")
        xr.testing.assert_allclose(ec.ecindex_smooth, expected, rtol=1e-3)

    def test_compute_pcs(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_pcs.nc")
        xr.testing.assert_allclose(ec.clim_pcs, expected, rtol=1e-3)

    def test_compute_pcs_projection(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_pcs_projection.nc")
        xr.testing.assert_allclose(ec.pcs, expected, rtol=1e-3)

    def test_compute_pcs_projection_smooth(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_pcs_projection_smooth.nc")
        xr.testing.assert_allclose(ec.pcs_smooth, expected, rtol=1e-3)

    def test_compute_eofs(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_eofs.nc")
        xr.testing.assert_allclose(ec.eofs, expected, rtol=1e-3)

    def test_compute_patterns(self, ec):
        expected = xr.open_dataset("tests/data/ecindex_patterns.nc")
        xr.testing.assert_allclose(ec.patterns, expected, rtol=1e-1)

    def test_custom_corr_factor(self, ersstv5):
        ec = xenso.indices.ECindex(
            ersstv5,
            base_period=("1991", "2020"),
            corr_factor=[-1, -1],
        )
        expected = xr.DataArray(
            [-1, -1],
            coords=[("mode", [1, 2])],
        )
        xr.testing.assert_allclose(ec.corr_factor, expected, rtol=1e-3)

    def test_custom_smooth_kernel(self, ersstv5):
        ec = xenso.indices.ECindex(
            ersstv5,
            base_period=("1991", "2020"),
            smooth_kernel=[1, 1, 1],
        )
        expected = xr.DataArray([1 / 3, 1 / 3, 1 / 3], dims=["time"])
        xr.testing.assert_allclose(ec.smooth_kernel, expected, rtol=1e-3)

    def test_compute_alpha(self):
        actual = xenso.indices.ECindex.compute_alpha(
            np.arange(1, 13),
            0.5 * np.arange(1, 13) ** 2,
        )
        expected = 0.5
        np.testing.assert_allclose(actual, expected)

    def test_compute_alpha_fit(self):
        actual_coef, actual_xfit, actual_fit = xenso.indices.ECindex.compute_alpha(
            np.arange(1, 13),
            0.5 * np.arange(1, 13) ** 2,
            return_fit=True,
        )
        expected_coef = 0.5
        expected_xfit = np.arange(1, 12 + 0.1, 0.1)
        expected_fit = expected_coef * expected_xfit**2
        np.testing.assert_allclose(actual_coef, expected_coef)
        np.testing.assert_allclose(actual_xfit, expected_xfit)
        np.testing.assert_allclose(actual_fit, expected_fit)


class TestENzones:
    @pytest.fixture(scope="class")
    def dummy(self):
        return xr.DataArray(
            np.tile(np.arange(360), (180, 1)),
            dims=["lat", "lon"],
            coords={"lat": np.arange(-90, 90), "lon": np.arange(360)},
        )

    @pytest.mark.parametrize(
        "zone,expected",
        [("12", 275), ("3", 240), ("34", 215), ("4", 185)],
    )
    def test_compute_enzones(self, dummy, zone, expected):
        actual = xenso.indices.enzones(dummy, zone=zone)
        np.testing.assert_allclose(actual, expected)
