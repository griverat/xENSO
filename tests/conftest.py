import pytest
import xarray as xr


@pytest.fixture(scope="module")
def ersstv5():
    return xr.open_dataarray("tests/data/ersstv5.nc")
