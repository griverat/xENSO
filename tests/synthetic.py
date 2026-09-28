"""Synthetic fields with known ENSO diagnostics, shared by the tests."""

import numpy as np
import pandas as pd
import xarray as xr

import xenso

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


def uniform_field(series, units=None, lat=LAT, lon=LON):
    field = series.expand_dims(lat=lat, lon=lon).transpose("time", "lat", "lon")
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
