import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import signal, stats

import xenso
from xenso.diagnostics import _SEAWATER_HEAT_CAPACITY, _SECONDS_PER_MONTH

N_YEARS = 25
# events in the first 24 years, the last year completes the last event
N_EVENTS = 24
TIME = pd.date_range("1980-01-01", periods=N_YEARS * 12, freq="MS")
LAT = np.arange(-10.0, 11.0, 2.0)
LON = np.arange(140.0, 286.0, 1.0)
# event shape around December (lags -3..3), and amplitudes with zero mean in
# every calendar month and no linear trend, so that anomalies and detrending
# leave the series unchanged
SHAPE = np.array([0.1, 0.4, 0.8, 1.0, 0.8, 0.4, 0.1])
AMPLITUDES = np.tile([1.0, -1.0, -1.0, 1.0], N_EVENTS // 4)


def event_series(amplitudes=AMPLITUDES, shape=SHAPE):
    values = np.zeros(N_YEARS * 12)
    half = len(shape) // 2
    for year, amplitude in enumerate(amplitudes):
        december = year * 12 + 11
        segment = slice(max(0, december - half), min(len(values), december + half + 1))
        offset = max(0, half - december)
        values[segment] += amplitude * shape[offset : offset + segment.stop - segment.start]
    return xr.DataArray(values, coords=[("time", TIME)])


def uniform_field(series, units=None):
    field = series.expand_dims(lat=LAT, lon=LON).transpose("time", "lat", "lon")
    if units is not None:
        field.attrs["units"] = units
    return field


def in_box(field, region):
    box = xenso.REGIONS[region]
    return (
        (field.lat >= box["lat"].start)
        & (field.lat <= box["lat"].stop)
        & (field.lon >= box["lon"].start)
        & (field.lon <= box["lon"].stop)
    )


def with_seasonal_cycle(field):
    return field + 26 + 2 * np.cos(2 * np.pi * field.time.dt.month / 12)


def reference_anomalies(series):
    """Anomalies from the full-record climatology and scipy.signal.detrend."""
    values = series.values.reshape(-1, 12)
    anomalies = (values - values.mean(axis=0)).ravel()
    return signal.detrend(anomalies) + anomalies.mean()


@pytest.fixture(scope="module")
def noise():
    rng = np.random.default_rng(42)
    return xr.DataArray(rng.gamma(2.0, size=N_YEARS * 12), coords=[("time", TIME)])


class TestScalarDiagnostics:
    def test_amplitude(self, noise):
        result = xenso.enso_amplitude(uniform_field(noise, units="degC"))
        expected = reference_anomalies(noise).std()
        np.testing.assert_allclose(result.value, expected)
        np.testing.assert_allclose(result.error, expected / np.sqrt(N_YEARS))
        assert result.value.attrs == {"units": "degC"}
        assert result.attrs["clivar_name"] == "EnsoAmpl"

    def test_amplitude_ignores_seasonal_cycle(self, noise):
        field = uniform_field(noise)
        months = field.time.dt.month
        np.testing.assert_allclose(
            xenso.enso_amplitude(field + 3 * np.sin(2 * np.pi * months / 12)).value,
            xenso.enso_amplitude(field).value,
        )

    def test_amplitude_uses_region(self, noise):
        field = uniform_field(noise)
        field = field.where(in_box(field, "nino3"), 10 * field)
        np.testing.assert_allclose(
            xenso.enso_amplitude(field, region="nino3").value,
            xenso.enso_amplitude(uniform_field(noise)).value,
        )

    def test_seasonality(self, noise):
        anomalies = reference_anomalies(noise).reshape(-1, 12)
        mam = anomalies[:, 2:5].mean(axis=1).std()
        ndj = np.stack([anomalies[:-1, 10], anomalies[:-1, 11], anomalies[1:, 0]]).mean(axis=0).std()
        result = xenso.enso_seasonality(uniform_field(noise))
        np.testing.assert_allclose(result.value, ndj / mam)
        np.testing.assert_allclose(result.ndj_std, ndj)
        ndj_err, mam_err = ndj / np.sqrt(N_YEARS - 1), mam / np.sqrt(N_YEARS)
        np.testing.assert_allclose(result.error, (mam * ndj_err + ndj * mam_err) / mam**2)

    def test_skewness(self, noise):
        result = xenso.enso_skewness(uniform_field(noise))
        expected = stats.skew(reference_anomalies(noise))
        np.testing.assert_allclose(result.value, expected)
        np.testing.assert_allclose(result.error, expected / np.sqrt(N_YEARS))
        assert result.value > 0

    def test_vectorized(self, noise):
        field = xr.concat([uniform_field(noise), 2 * uniform_field(noise)], dim="member")
        np.testing.assert_allclose(
            xenso.enso_amplitude(field).value[1], 2 * xenso.enso_amplitude(field).value[0]
        )


class TestDuration:
    def test_without_smoothing(self):
        result = xenso.enso_duration(uniform_field(event_series()), smoothing=None)
        # the lifecycle is the event shape: above 0.25 from lag -2 to lag 2
        np.testing.assert_allclose(result.lifecycle.sel(lag=slice(-3, 3)), SHAPE)
        assert result.value == 5
        assert result.value.attrs["units"] == "months"

    def test_with_smoothing(self):
        result = xenso.enso_duration(with_seasonal_cycle(uniform_field(event_series())))
        smoothed = np.convolve(SHAPE, [1, 2, 3, 2, 1], mode="full") / 9
        smoothed = smoothed / smoothed.max()
        np.testing.assert_allclose(result.lifecycle.sel(lag=slice(-5, 5)), smoothed, atol=1e-3)
        assert result.value == (smoothed > 0.25).sum()

    def test_threshold(self):
        result = xenso.enso_duration(uniform_field(event_series()), threshold=0.9, smoothing=None)
        assert result.value == 1


class TestDiversity:
    PEAKS = {3: 250.0, 7: 200.0, 11: 230.0, 15: 190.0, 19: 210.0}  # event year index -> longitude
    SIGNS = {3: 1, 7: -1, 11: 1, 15: -1, 19: 1}

    @pytest.fixture(scope="class")
    def field(self):
        amplitude = xr.zeros_like(uniform_field(event_series()))
        for year, lon in self.PEAKS.items():
            amplitudes = np.zeros(N_EVENTS)
            amplitudes[year] = 3.0 * self.SIGNS[year]
            profile = np.exp(-(((LON - lon) / 30.0) ** 2))
            amplitude = amplitude + event_series(amplitudes) * xr.DataArray(
                profile, coords=[("lon", LON)]
            )
        return amplitude

    def test_peak_longitudes(self, field):
        result = xenso.enso_diversity(field)
        years = [TIME[0].year + year for year in self.PEAKS]
        np.testing.assert_array_equal(result.peak_lon.year, years)
        # removing the climatology shifts the peaks slightly, as in CLIVAR
        np.testing.assert_allclose(result.peak_lon, list(self.PEAKS.values()), atol=2)
        kinds = ["nino" if sign > 0 else "nina" for sign in self.SIGNS.values()]
        np.testing.assert_array_equal(result.peak_lon.kind, kinds)

    def test_dispersion(self, field):
        result = xenso.enso_diversity(field)
        lons = result.peak_lon.values
        np.testing.assert_allclose(result.value, np.percentile(lons, 75) - np.percentile(lons, 25))
        np.testing.assert_allclose(result.mad, np.median(np.abs(lons - np.median(lons))))
        assert result.attrs["clivar_name"] == "EnsoSstDiversity"


class TestFeedbacks:
    @pytest.fixture(scope="class")
    def series(self):
        return event_series()

    def two_boxes(self, series, region, factor):
        """Field equal to factor * series in region and 100 * series elsewhere."""
        field = uniform_field(series)
        return with_seasonal_cycle(field.where(in_box(field, region), 100 * field) * factor)

    @pytest.mark.parametrize(
        "function,response,forcing,expected",
        [
            (xenso.bjerknes_feedback, ("taux", "nino4", 3.0), ("sst", "nino3", 1.0), 3.0),
            (xenso.heat_flux_feedback, ("thf", "nino3", -20.0), ("sst", "nino3", 1.0), -20.0),
            (xenso.thermocline_feedback, ("sst", "nino3", 0.5), ("ssh", "nino3", 1.0), 0.5),
            (xenso.wind_ssh_feedback, ("ssh", "nino3", 4.0), ("taux", "nino4", 1.0), 4.0),
        ],
    )
    def test_slope_and_regions(self, series, function, response, forcing, expected):
        fields = {
            name: self.two_boxes(series, region, factor) for name, region, factor in [response, forcing]
        }
        result = function(**fields)
        np.testing.assert_allclose(result.value, expected)
        np.testing.assert_allclose(result.error, 0, atol=1e-6)
        np.testing.assert_allclose(result.slope, expected)
        np.testing.assert_allclose(result.nonlinearity, 0, atol=1e-6)

    def test_units_and_metadata(self, series):
        sst = uniform_field(series, units="degC")
        taux = uniform_field(2 * series, units="N m-2")
        result = xenso.bjerknes_feedback(sst, taux)
        assert result.value.attrs["units"] == "N m-2 / (degC)"
        assert result.attrs["clivar_name"] == "EnsoFbSstTaux"
        assert "units" not in xenso.bjerknes_feedback(uniform_field(series), taux).value.attrs

    def test_different_periods_are_aligned(self, series):
        sst = uniform_field(series)
        result = xenso.bjerknes_feedback(sst, 2 * sst.isel(time=slice(48, None)))
        assert np.isfinite(result.value)


class TestOceanDrivenSstChange:
    TO_TEMPERATURE = _SECONDS_PER_MONTH / (_SEAWATER_HEAT_CAPACITY * 50.0)

    @pytest.fixture(scope="class")
    def sst(self):
        return uniform_field(event_series())

    def test_no_heat_flux(self, sst):
        result = xenso.ocean_driven_sst_change(sst, xr.zeros_like(sst))
        np.testing.assert_allclose(result.value, 1)
        np.testing.assert_array_equal(result.month, np.arange(6, 13))
        np.testing.assert_allclose(result.dsst.sel(month=[6, 12]), [0, 1])
        np.testing.assert_allclose(result.dsst_thf, 0)

    def test_heat_flux_explains_all(self, sst):
        # the heat flux that produces the monthly SST tendency of a 50 m slab
        tendency = sst.diff("time", label="upper").reindex(time=sst.time, fill_value=0.0)
        result = xenso.ocean_driven_sst_change(sst, tendency / self.TO_TEMPERATURE)
        np.testing.assert_allclose(result.value, 0, atol=1e-8)
        np.testing.assert_allclose(result.dsst_thf, result.dsst)

    def test_min_change(self, sst):
        result = xenso.ocean_driven_sst_change(sst, xr.zeros_like(sst), min_change=10)
        assert result.value.isnull()


class TestObservations:
    """Sanity checks on ERSSTv5 (1990-2023) against the observed CLIVAR ranges."""

    def test_amplitude(self, ersstv5):
        assert 0.6 < xenso.enso_amplitude(ersstv5).value < 1.1

    def test_seasonality_and_skewness(self, ersstv5):
        assert xenso.enso_seasonality(ersstv5).value > 1.5
        assert xenso.enso_skewness(ersstv5).value > 0

    def test_duration(self, ersstv5):
        assert 8 <= xenso.enso_duration(ersstv5).value <= 16

    def test_diversity(self, ersstv5):
        result = xenso.enso_diversity(ersstv5)
        assert 10 < result.value < 60
        assert result.peak_lon.sel(year=1997) > result.peak_lon.sel(year=2002)

    def test_compare(self, ersstv5):
        early = xenso.enso_amplitude(ersstv5.sel(time=slice(None, "2009")))
        value, error = xenso.compare(early, xenso.enso_amplitude(ersstv5))
        assert value > 0
        assert error > 0
