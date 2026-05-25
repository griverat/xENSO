import numpy as np
import xarray as xr

from xenso.preprocessing import normalize_coords


def make_grid(lat_ascending=True, lon_ascending=True):
    lats = np.arange(-90, 91, 10)
    lons = np.arange(0, 360, 10)
    if not lat_ascending:
        lats = lats[::-1]
    if not lon_ascending:
        lons = lons[::-1]
    return xr.DataArray(
        np.zeros((len(lats), len(lons))),
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
    )


def test_ascending_lat_unchanged():
    data = make_grid(lat_ascending=True)
    result = normalize_coords(data)
    xr.testing.assert_equal(result, data)


def test_descending_lat_sorted():
    data = make_grid(lat_ascending=False)
    result = normalize_coords(data)
    assert (result.lat.values == sorted(result.lat.values)).all()


def test_descending_lon_sorted():
    data = make_grid(lon_ascending=False)
    result = normalize_coords(data)
    assert (result.lon.values == sorted(result.lon.values)).all()


def test_both_descending_sorted():
    data = make_grid(lat_ascending=False, lon_ascending=False)
    result = normalize_coords(data)
    assert (result.lat.values == sorted(result.lat.values)).all()
    assert (result.lon.values == sorted(result.lon.values)).all()


def test_negative_lon_converted_to_0_360():
    lons = np.array([-180, -90, 0, 90, 180])
    data = xr.DataArray(np.zeros(5), dims=["lon"], coords={"lon": lons})
    result = normalize_coords(data)
    assert (result.lon.values >= 0).all()
    assert (result.lon.values <= 360).all()


def test_negative_lon_values_match_0_360():
    # -180 → 180, -90 → 270, 0 → 0, 90 → 90
    lons = np.array([-180, -90, 0, 90])
    data = xr.DataArray(np.arange(4), dims=["lon"], coords={"lon": lons})
    result = normalize_coords(data)
    np.testing.assert_array_equal(result.lon.values, [0, 90, 180, 270])


def test_already_0_360_lon_unchanged():
    lons = np.arange(0, 360, 10)
    data = make_grid(lon_ascending=True)
    result = normalize_coords(data)
    np.testing.assert_array_equal(result.lon.values, lons)


def test_no_lat_lon_dims_unchanged():
    data = xr.DataArray(np.arange(10), dims=["time"])
    result = normalize_coords(data)
    xr.testing.assert_equal(result, data)
