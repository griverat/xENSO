import numpy as np
import pytest
import xarray as xr
from scipy import stats

import xenso


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 * x + 0.3 * np.where(x < 0, x, 0) + rng.normal(scale=0.2, size=200)
    return xr.DataArray(x, dims="time"), xr.DataArray(y, dims="time")


class TestLinregress:
    @pytest.mark.parametrize("sign", [None, "positive", "negative"])
    def test_matches_scipy(self, xy, sign):
        x, y = xy
        mask = {None: np.ones(x.size, bool), "positive": x > 0, "negative": x < 0}[sign]
        expected = stats.linregress(x.values[mask], y.values[mask])
        result = xenso.linregress(y, x, sign=sign)
        np.testing.assert_allclose(result.slope, expected.slope)
        np.testing.assert_allclose(result.intercept, expected.intercept)
        np.testing.assert_allclose(result.stderr, expected.stderr)

    def test_skips_missing(self, xy):
        x, y = xy
        y_gaps = y.where(np.arange(y.size) % 7 != 0)
        keep = np.arange(y.size) % 7 != 0
        expected = stats.linregress(x.values[keep], y.values[keep])
        np.testing.assert_allclose(xenso.linregress(y_gaps, x).slope, expected.slope)

    def test_map_on_index(self, xy):
        x, y = xy
        field = xr.concat([y, 2 * y, y.where(False)], dim="lon")
        result = xenso.linregress(field, x)
        assert result.slope.dims == ("lon",)
        np.testing.assert_allclose(result.slope[1], 2 * result.slope[0])
        assert result.slope[2].isnull()

    def test_perfect_fit(self, xy):
        x, _ = xy
        result = xenso.linregress(x, x)
        np.testing.assert_allclose(result.slope, 1)
        np.testing.assert_allclose(result.stderr, 0, atol=1e-12)

    def test_invalid_sign(self, xy):
        with pytest.raises(ValueError):
            xenso.linregress(*xy, sign="zero")


def test_split_regression(xy):
    x, y = xy
    result = xenso.split_regression(y, x)
    np.testing.assert_array_equal(result.branch, ["all", "positive", "negative"])
    for branch, sign in [("all", None), ("positive", "positive"), ("negative", "negative")]:
        expected = xenso.linregress(y, x, sign=sign)
        np.testing.assert_allclose(result.slope.sel(branch=branch), expected.slope)
    slopes = result.slope
    np.testing.assert_allclose(
        result.nonlinearity, slopes.sel(branch="negative") - slopes.sel(branch="positive")
    )
    # y responds more strongly to negative x
    np.testing.assert_allclose(result.nonlinearity, 0.3, atol=0.1)
    assert "branch" not in result.nonlinearity_stderr.coords


class TestRmse:
    @pytest.fixture(scope="class")
    def fields(self):
        model = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="lon")
        obs = xr.DataArray([0.0, 1.0, 3.0, 2.0], dims="lon")
        return model, obs

    def test_unweighted(self, fields):
        np.testing.assert_allclose(xenso.rmse(*fields, dim="lon"), np.sqrt(6 / 4))

    def test_weighted(self, fields):
        weights = xr.DataArray([0.0, 0.0, 1.0, 1.0], dims="lon")
        np.testing.assert_allclose(xenso.rmse(*fields, dim="lon", weights=weights), np.sqrt(2))

    def test_centered(self, fields):
        model, obs = fields
        np.testing.assert_allclose(xenso.rmse(obs + 5, obs, dim="lon", centered=True), 0)
        np.testing.assert_allclose(
            xenso.rmse(model, obs, dim="lon", centered=True), np.std([1, 1, 0, 2])
        )

    def test_skips_missing(self, fields):
        model, obs = fields
        result = xenso.rmse(model.where(model != 4), obs, dim="lon")
        np.testing.assert_allclose(result, np.sqrt(2 / 3))


class TestCompare:
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("difference", 0.3),
            ("ratio", 4 / 3),
            ("relative_difference", 1 / 3),
            ("abs_relative_difference", 100 / 3),
        ],
    )
    def test_value(self, method, expected):
        value, error = xenso.compare(1.2, 0.9, method=method)
        np.testing.assert_allclose(value, expected)
        assert error is None

    def test_abs_relative_difference_is_symmetric_in_sign(self):
        np.testing.assert_allclose(xenso.compare(0.6, 0.9)[0], 100 / 3)

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("difference", 0.15),
            ("ratio", (0.9 * 0.1 + 1.2 * 0.05) / 0.81),
            ("relative_difference", (0.9 * 0.1 + 1.2 * 0.05) / 0.81),
            ("abs_relative_difference", 100 * (0.9 * 0.1 + 1.2 * 0.05) / 0.81),
        ],
    )
    def test_error(self, method, expected):
        _, error = xenso.compare(1.2, 0.9, method=method, model_err=0.1, obs_err=0.05)
        np.testing.assert_allclose(error, expected)

    def test_dataarray(self):
        model = xr.DataArray([1.0, 2.0], dims="model")
        value, _ = xenso.compare(model, 1.0, method="ratio")
        xr.testing.assert_allclose(value, model)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            xenso.compare(1.0, 1.0, method="distance")


def test_linregress_without_intercept(xy):
    x, y = xy
    result = xenso.linregress(y, x, fit_intercept=False)
    slope = (x * y).sum() / (x**2).sum()
    residual = ((y - slope * x) ** 2).sum()
    np.testing.assert_allclose(result.slope, slope)
    np.testing.assert_allclose(result.intercept, 0)
    np.testing.assert_allclose(result.stderr, np.sqrt(residual / (x.size - 1) / (x**2).sum()))


def test_compare_datasets():
    model = xr.Dataset({"value": 1.2, "error": 0.1})
    obs = xr.Dataset({"value": 0.9, "error": 0.05})
    value, error = xenso.compare(model, obs, method="difference")
    np.testing.assert_allclose([value, error], [0.3, 0.15])
    assert xenso.compare(xr.Dataset({"value": 1.0}), obs)[1] is None
