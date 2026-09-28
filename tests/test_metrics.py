import numpy as np
import pytest
import xarray as xr
from synthetic import LON, event_series, uniform_field, with_seasonal_cycle

import xenso
from xenso.metrics import _TELECONNECTION_EXCLUDED, _outside_box

GLOBAL_LAT = np.arange(-30.0, 31.0, 5.0)
GLOBAL_LON = np.arange(0.0, 360.0, 5.0)


@pytest.fixture(scope="module")
def series():
    return event_series()


def lon_profile(lon=LON):
    return xr.DataArray(1 + np.sin(np.deg2rad(lon)), coords=[("lon", lon)])


class TestMeanState:
    @pytest.fixture(scope="class")
    def obs(self, series):
        return with_seasonal_cycle(uniform_field(series, units="degC") + lon_profile())

    @pytest.mark.parametrize("along", ["lon", "lat"])
    def test_constant_bias(self, obs, along):
        result = xenso.mean_state_rmse(obs + 0.5, obs, along=along)
        np.testing.assert_allclose(result.value, 0.5)
        assert result.model.dims == (along,)
        xr.testing.assert_allclose(result.model - result.obs, xr.full_like(result.model, 0.5))

    def test_section(self, obs):
        result = xenso.mean_state_rmse(obs + lon_profile(), obs, variable="sst")
        section = lon_profile().sel(lon=slice(150, 270))
        np.testing.assert_allclose(result.value, np.sqrt((section**2).mean()))
        assert result.attrs["clivar_name"] == "BiasSstLonRmse"
        assert result.value.attrs["units"] == "degC"

    def test_only_region_matters(self, obs):
        model = obs.where(obs.lon <= 270, obs + 10)
        np.testing.assert_allclose(xenso.mean_state_rmse(model, obs).value, 0)

    def test_different_grids(self, obs):
        with pytest.raises(ValueError, match="same grid"):
            xenso.mean_state_rmse(obs, obs.isel(lon=slice(None, None, 2)))
        # tiny coordinate differences are accepted
        shifted = obs.assign_coords(lat=obs.lat + 1e-10)
        np.testing.assert_allclose(xenso.mean_state_rmse(shifted, obs, along="lat").value, 0)


class TestSeasonalCycle:
    def cycle(self, amplitude):
        field = xr.zeros_like(uniform_field(event_series()))
        return field + 26 + amplitude * np.cos(2 * np.pi * field.time.dt.month / 12)

    @pytest.mark.parametrize("along", ["lon", "lat"])
    def test_amplitude_difference(self, along):
        result = xenso.seasonal_cycle_rmse(self.cycle(3.0), self.cycle(1.0), along=along, variable="pr")
        # the standard deviation of a cosine over 12 months is amplitude / sqrt(2)
        np.testing.assert_allclose(result.value, 2 / np.sqrt(2), rtol=1e-2)
        np.testing.assert_allclose(result.obs, 1 / np.sqrt(2), rtol=1e-2)
        assert result.attrs["clivar_name"] == f"SeasonalPr{along.capitalize()}Rmse"

    def test_ignores_interannual_signal(self, series):
        field = self.cycle(1.0)
        result = xenso.seasonal_cycle_rmse(field + uniform_field(series), field)
        # the events have zero mean in every calendar month
        np.testing.assert_allclose(result.value, 0, atol=1e-10)


class TestEnsoPattern:
    def test_exact_pattern(self, series):
        field = uniform_field(series) * lon_profile()
        result = xenso.enso_pattern_rmse(field, field)
        nino34_mean = lon_profile().sel(lon=slice(190, 240)).mean()
        expected = (lon_profile() / nino34_mean).sel(lon=slice(150, 270))
        np.testing.assert_allclose(result.obs, expected)
        np.testing.assert_allclose(result.value, 0)
        assert result.attrs["clivar_name"] == "EnsoSstLonRmse"

    def test_scale_invariant(self, series):
        field = uniform_field(series) * lon_profile()
        np.testing.assert_allclose(xenso.enso_pattern_rmse(3 * field, field).value, 0, atol=1e-12)

    def test_other_field(self, series):
        sst = uniform_field(series) * lon_profile()
        result = xenso.enso_pattern_rmse(sst, sst, 3 * sst, sst, variable="taux")
        np.testing.assert_allclose(result.model, 3 * result.obs)
        np.testing.assert_allclose(result.value, 2 * np.sqrt((result.obs**2).mean()))
        assert result.attrs["clivar_name"] == "EnsoTauxLonRmse"


class TestEnsoLifecycle:
    def test_matches_duration_lifecycles(self, series):
        obs = uniform_field(series)
        model = uniform_field(event_series(shape=np.array([0.2, 0.5, 0.9, 1.0, 0.9, 0.5, 0.2])))
        result = xenso.enso_lifecycle_rmse(model, obs)
        expected = xenso.rmse(
            xenso.enso_duration(model).lifecycle, xenso.enso_duration(obs).lifecycle, dim="lag"
        )
        np.testing.assert_allclose(result.value, expected)
        assert result.value > 0
        assert result.model.dims == ("lag",)

    def test_grids_can_differ(self, series):
        obs = uniform_field(series)
        result = xenso.enso_lifecycle_rmse(obs.isel(lon=slice(None, None, 2)), obs)
        np.testing.assert_allclose(result.value, 0, atol=1e-12)


class TestTeleconnection:
    @pytest.fixture(scope="class")
    def pattern(self):
        lat = xr.DataArray(GLOBAL_LAT, coords=[("lat", GLOBAL_LAT)])
        lon = xr.DataArray(GLOBAL_LON, coords=[("lon", GLOBAL_LON)])
        return np.cos(np.deg2rad(lat)) * (1.5 + np.sin(2 * np.deg2rad(lon)))

    @pytest.fixture(scope="class")
    def sst(self, series):
        return uniform_field(series, lat=GLOBAL_LAT, lon=GLOBAL_LON)

    def test_exact_maps(self, sst, pattern):
        field = sst * pattern
        result = xenso.enso_teleconnection(sst, sst, field, field, variable="pr")
        # the index SST is uniform, so the regression map is the pattern itself
        expected = pattern.where(_outside_box(pattern, _TELECONNECTION_EXCLUDED))
        xr.testing.assert_allclose(result.obs, expected.transpose(*result.obs.dims))
        np.testing.assert_allclose([result.value, result.correlation, result.std_ratio], [0, 1, 1])
        assert result.attrs["clivar_name"] == "EnsoPrMapDjf"

    def test_default_mask(self, sst, pattern):
        result = xenso.enso_teleconnection(sst, sst, sst * pattern, sst * pattern)
        inside = (abs(result.obs.lat) < 15) & (result.obs.lon > 120) & (result.obs.lon < 285)
        assert (result.obs.notnull() == ~inside).all()

    def test_scaled_model(self, sst, pattern):
        field = sst * pattern
        result = xenso.enso_teleconnection(sst, sst, 2 * field, field)
        obs = result.obs
        np.testing.assert_allclose(result.correlation, 1)
        np.testing.assert_allclose(result.std_ratio, 2)
        np.testing.assert_allclose(result.value, np.sqrt((obs**2).mean()))

    def test_keep_and_weights(self, sst, pattern):
        field = sst * pattern
        noise = xr.DataArray(np.random.default_rng(0).normal(size=pattern.shape), coords=pattern.coords)
        keep = xr.ones_like(pattern, dtype=bool)
        result = xenso.enso_teleconnection(sst, sst, field + sst * noise, field, keep=keep)
        assert result.obs.notnull().all()
        weighted = xenso.enso_teleconnection(
            sst, sst, field + sst * noise, field, keep=keep, weighted=True
        )
        diff = result.model - result.obs
        weights = np.cos(np.deg2rad(diff.lat))
        np.testing.assert_allclose(weighted.value, np.sqrt((diff**2).weighted(weights).mean()))
        assert weighted.value != result.value

    def test_different_grids(self, sst):
        with pytest.raises(ValueError, match="same grid"):
            xenso.enso_teleconnection(sst, sst.isel(lat=slice(1, None)))


class TestObservations:
    """Comparing two halves of ERSSTv5 gives small errors and similar patterns."""

    @pytest.fixture(scope="class")
    def halves(self, ersstv5):
        return ersstv5.sel(time=slice(None, "2006")), ersstv5.sel(time=slice("2007", None))

    def test_sections(self, halves):
        assert xenso.mean_state_rmse(*halves).value < 0.5
        assert xenso.seasonal_cycle_rmse(*halves).value < 0.2

    def test_enso(self, halves):
        assert xenso.enso_pattern_rmse(*halves).value < 0.3
        assert xenso.enso_lifecycle_rmse(*halves).value < 0.3
        assert xenso.enso_teleconnection(*halves).correlation > 0.5
