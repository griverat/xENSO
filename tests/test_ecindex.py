import numpy as np
import pytest
import xarray as xr

import xenso


class TestECindex:
    @pytest.fixture(scope="class")
    def ec(self, ersstv5):
        return xenso.ECindex(ersstv5, base_period=("1991", "2020"))

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
        ec = xenso.ECindex(ersstv5, base_period=("1991", "2020"), corr_factor=[-1, -1])
        expected = xr.DataArray([-1, -1], coords=[("mode", [1, 2])])
        xr.testing.assert_allclose(ec.corr_factor, expected, rtol=1e-3)

    def test_custom_smooth_kernel(self, ersstv5):
        ec = xenso.ECindex(
            ersstv5,
            base_period=("1991", "2020"),
            smooth_kernel=[1, 1, 1],
        )
        expected = xr.DataArray([1 / 3, 1 / 3, 1 / 3], dims=["time"])
        xr.testing.assert_allclose(ec.smooth_kernel, expected, rtol=1e-3)

    def test_isanomaly(self, ersstv5):
        anomaly = xenso.compute_anomaly(ersstv5, base_period=("1991", "2020"))
        ec = xenso.ECindex(anomaly, isanomaly=True, base_period=("1991", "2020"))
        assert ec.climatology is None

    def test_isanomaly_with_climatology(self, ersstv5):
        clim = xenso.compute_climatology(ersstv5, ("1991", "2020"))
        ec = xenso.ECindex(ersstv5, climatology=clim, base_period=("1991", "2020"))
        xr.testing.assert_equal(ec.climatology, clim)

    def test_compute_alpha(self):
        actual = xenso.ECindex.compute_alpha(
            np.arange(1, 13),
            0.5 * np.arange(1, 13) ** 2,
        )
        np.testing.assert_allclose(actual, 0.5)

    def test_compute_alpha_fit(self):
        coef, xfit, fit = xenso.ECindex.compute_alpha(
            np.arange(1, 13),
            0.5 * np.arange(1, 13) ** 2,
            return_fit=True,
        )
        expected_xfit = np.arange(1, 12 + 0.1, 0.1)
        np.testing.assert_allclose(coef, 0.5)
        np.testing.assert_allclose(xfit, expected_xfit)
        np.testing.assert_allclose(fit, 0.5 * expected_xfit**2)
