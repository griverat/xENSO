import pytest
import xarray as xr

import xenso


class TestECindex:
    @pytest.fixture(scope="class")
    def ec(self, ersstv5):
        return xenso.indices.ECindex(ersstv5, base_period=("1991", "2020"))

    def test_compute_ecindex(self, ec):
        expected = xr.open_dataset("tests/data/ecindex.nc")
        xr.testing.assert_allclose(ec.ecindex, expected)

    def test_compute_ecindex_smooth(self, ec):
        expected = xr.open_dataset("tests/data/ecindex_smooth.nc")
        xr.testing.assert_allclose(ec.ecindex_smooth, expected)

    def test_compute_pcs(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_pcs.nc")
        xr.testing.assert_allclose(ec.clim_pcs, expected)

    def test_compute_pcs_projection(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_pcs_projection.nc")
        xr.testing.assert_allclose(ec.pcs, expected)

    def test_compute_pcs_projection_smooth(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_pcs_projection_smooth.nc")
        xr.testing.assert_allclose(ec.pcs_smooth, expected)

    def test_compute_eofs(self, ec):
        expected = xr.open_dataarray("tests/data/ecindex_eofs.nc")
        xr.testing.assert_allclose(ec.eofs, expected)

    def test_compute_patterns(self, ec):
        expected = xr.open_dataset("tests/data/ecindex_patterns.nc")
        xr.testing.assert_allclose(ec.patterns, expected)
