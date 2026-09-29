"""
Model-observation ENSO metrics.

Each function computes one metric of the CLIVAR ENSO Metrics Package
(https://github.com/CLIVAR-PRP/enso_metrics), named in its docstring and in the
``clivar_name`` attribute of the result: the RMSE between a model and a
reference dataset of a mean-state, seasonal-cycle or ENSO pattern. The
result is a Dataset with the metric as ``value`` and the compared model and
observed profiles as ``model`` and ``obs``.

CLIVAR regrids both datasets to a 1° grid; here they must already share the
same grid, otherwise a ValueError is raised. Preprocessing follows
:mod:`xenso.diagnostics`. This is not the official implementation: use the
original package for results that must be comparable with published CLIVAR
ENSO metrics.
"""

from typing import Literal

import numpy as np
import xarray as xr

from .core import detrend
from .diagnostics import EnsoKind, _anomalies, _lifecycle, _prefix, _result
from .events import detect_events, seasonal_series
from .preprocessing import normalize_coords
from .regions import REGIONS
from .stats import linregress, rmse

# CLIVAR's teleconnection maps leave out the Pacific basin between 15°S and 15°N
_TELECONNECTION_EXCLUDED = {"lat": (-15.0, 15.0), "lon": (120.0, 285.0)}


def _box(data: xr.DataArray, region: str) -> xr.DataArray:
    return normalize_coords(data).sel(**REGIONS[region])


def _same_grid(model: xr.DataArray, obs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Raise unless both fields have the same lat/lon coordinates."""
    for dim in ("lat", "lon"):
        if dim not in model.dims:
            continue
        a, b = model[dim].values, obs[dim].values
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError(
                f"model and obs must be on the same grid, but their {dim} coordinates differ; "
                "regrid them to a common grid first"
            )
        obs = obs.assign_coords({dim: model[dim]})
    return model, obs


def _meridional_mean(data: xr.DataArray) -> xr.DataArray:
    return data.weighted(np.cos(np.deg2rad(data.lat))).mean("lat")


def _zonal_mean(data: xr.DataArray) -> xr.DataArray:
    return data.mean("lon")


def _section(data: xr.DataArray, along: Literal["lon", "lat"]) -> xr.DataArray:
    return _meridional_mean(data) if along == "lon" else _zonal_mean(data)


def _section_rmse(model, obs, along, clivar_name, units=None) -> xr.Dataset:
    model, obs = _section(model, along), _section(obs, along)
    return _result(rmse(model, obs, dim=along), clivar_name, units, model=model, obs=obs)


def _clivar_name(prefix: str, variable: str | None, suffix: str) -> str:
    return f"{prefix}{(variable or 'Var').capitalize()}{suffix}"


def mean_state_rmse(
    model: xr.DataArray,
    obs: xr.DataArray,
    along: Literal["lon", "lat"] = "lon",
    region: str | None = None,
    variable: str | None = None,
) -> xr.Dataset:
    """
    RMSE of the time-mean field along an equatorial section
    (CLIVAR: BiasSstLonRmse, BiasPrLonRmse, BiasTauxLonRmse, BiasPrLatRmse, ...).

    The time mean is averaged over latitude (cos-weighted) for a zonal
    section or over longitude for a meridional section, and the RMSE is
    taken along the section without weights.

    Parameters
    ----------
    model, obs
        Monthly fields on the same grid, with time, lat and lon dimensions.
    along
        "lon" for a zonal section, "lat" for a meridional section.
    region
        Box of :data:`xenso.REGIONS`. Defaults to "equatorial_pacific" for a
        zonal section and "nino3_latext" for a meridional one (CLIVAR uses
        "equatorial_pacific_latext" for zonal wind stress).
    variable
        Name used in ``clivar_name``, e.g. "sst" gives "BiasSstLonRmse".
    """
    region = region or ("equatorial_pacific" if along == "lon" else "nino3_latext")
    units = model.attrs.get("units")
    model, obs = _same_grid(_box(model, region).mean("time"), _box(obs, region).mean("time"))
    name = _clivar_name("Bias", variable, f"{along.capitalize()}Rmse")
    return _section_rmse(model, obs, along, name, units)


def seasonal_cycle_rmse(
    model: xr.DataArray,
    obs: xr.DataArray,
    along: Literal["lon", "lat"] = "lon",
    region: str | None = None,
    variable: str | None = None,
) -> xr.Dataset:
    """
    RMSE of the amplitude of the seasonal cycle along an equatorial section
    (CLIVAR: SeasonalSstLonRmse, SeasonalPrLatRmse, ...).

    The amplitude is the standard deviation of the 12 monthly means of the
    linearly detrended field at each grid point, which is then averaged as in
    :func:`mean_state_rmse`.

    Parameters are the same as :func:`mean_state_rmse`.
    """
    region = region or ("equatorial_pacific" if along == "lon" else "nino3_latext")

    def amplitude(data):
        return detrend(_box(data, region)).groupby("time.month").mean("time").std("month")

    units = model.attrs.get("units")
    model, obs = _same_grid(amplitude(model), amplitude(obs))
    name = _clivar_name("Seasonal", variable, f"{along.capitalize()}Rmse")
    return _section_rmse(model, obs, along, name, units)


def _enso_pattern(
    sst, field, index_region, field_region, season, smoothing, kind, threshold
) -> xr.DataArray:
    """
    ENSO pattern of the seasonal field anomaly: its regression, without intercept, onto the
    seasonal index ("enso"), or its composite over El Niño ("nino") or La Niña ("nina") events.
    """
    index = _anomalies(sst, index_region, smoothing)
    if field_region is not None:
        field = _box(field, field_region)
    anomalies = seasonal_series(_anomalies(field, smoothing=smoothing), season)
    anomalies = anomalies - anomalies.mean("year")
    if kind == "enso":
        peak = seasonal_series(index, season)
        peak = peak - peak.mean("year")
        return linregress(anomalies, peak, dim="year", fit_intercept=False).slope
    events = detect_events(index, threshold=threshold, season=season, kind=kind)
    return anomalies.sel(year=events.year).mean("year")


def enso_pattern_rmse(
    model_sst: xr.DataArray,
    obs_sst: xr.DataArray,
    model_field: xr.DataArray | None = None,
    obs_field: xr.DataArray | None = None,
    region: str = "equatorial_pacific",
    index_region: str = "nino34",
    season: str | int = "DEC",
    smoothing: int | None = 5,
    variable: str | None = "sst",
    kind: EnsoKind = "enso",
    threshold: float = 0.75,
) -> xr.Dataset:
    """
    RMSE of the zonal pattern of ENSO along the equator
    (CLIVAR: EnsoSstLonRmse, NinoSstLonRmse, NinaSstLonRmse, and EnsoPrLonRmse
    or EnsoTauxLonRmse with another field).

    The seasonal anomaly of the field, averaged over latitude, is regressed
    without intercept onto the seasonal Niño 3.4 SST anomaly ("enso"), or
    averaged over the El Niño ("nino") or La Niña ("nina") events of each
    dataset, and the RMSE of the resulting profiles is taken along longitude.

    Parameters
    ----------
    model_sst, obs_sst
        Monthly SST used for the ENSO index.
    model_field, obs_field
        Monthly fields whose pattern is compared, on the same grid. Default
        to the SST.
    region
        Box of :data:`xenso.REGIONS` of the pattern.
    index_region
        Box of :data:`xenso.REGIONS` of the index.
    season
        Month or season of the index and the pattern.
    smoothing
        Length of the triangular running mean in time, or None.
    variable
        Name used in ``clivar_name``.
    kind
        "enso" for the regression, "nino" or "nina" for event composites.
    threshold
        Event threshold in standard deviations, for "nino" and "nina".
    """
    prefix = _prefix(kind)
    model_field = model_sst if model_field is None else model_field
    obs_field = obs_sst if obs_field is None else obs_field

    def pattern(sst, field):
        section = _meridional_mean(_box(field, region))
        return _enso_pattern(sst, section, index_region, None, season, smoothing, kind, threshold)

    model, obs = _same_grid(pattern(model_sst, model_field), pattern(obs_sst, obs_field))
    name = _clivar_name(prefix, variable, "LonRmse")
    return _result(rmse(model, obs, dim="lon"), name, model=model, obs=obs)


def enso_lifecycle_rmse(
    model_sst: xr.DataArray,
    obs_sst: xr.DataArray,
    model_field: xr.DataArray | None = None,
    obs_field: xr.DataArray | None = None,
    region: str = "nino34",
    field_region: str | None = None,
    month: int = 12,
    window: int = 6,
    smoothing: int | None = 5,
    variable: str | None = "sst",
    kind: EnsoKind = "enso",
    threshold: float = 0.75,
) -> xr.Dataset:
    """
    RMSE of the ENSO life cycle (CLIVAR: EnsoSstTsRmse, NinoSstTsRmse,
    NinaSstTsRmse, and EnsoPrTsRmse or EnsoTauxTsRmse with another field).

    The life cycle is the monthly anomaly of the field, averaged over
    ``field_region``, around each year: regressed without intercept onto the
    Niño 3.4 SST anomaly in ``month`` ("enso", as in :func:`xenso.enso_duration`),
    or averaged over the El Niño ("nino") or La Niña ("nina") events of each
    dataset. The RMSE is taken along the lag.

    Parameters
    ----------
    model_sst, obs_sst
        Monthly SST with time, lat and lon dimensions. They do not need to
        share a grid.
    model_field, obs_field
        Monthly fields whose life cycle is compared. Default to the SST.
        CLIVAR uses precipitation in "nino3" and zonal wind stress in "nino4".
    region
        Box of :data:`xenso.REGIONS` of the index.
    field_region
        Box of :data:`xenso.REGIONS` of the field. Defaults to ``region``.
    month, window, smoothing
        See :func:`xenso.enso_duration`.
    variable
        Name used in ``clivar_name``.
    kind
        "enso" for the regression, "nino" or "nina" for event composites.
    threshold
        Event threshold in standard deviations, for "nino" and "nina".
    """
    prefix = _prefix(kind)
    kwargs = dict(
        region=region,
        field_region=field_region,
        month=month,
        window=window,
        smoothing=smoothing,
        kind=kind,
        threshold=threshold,
    )
    model = _lifecycle(model_sst, model_field, **kwargs)
    obs = _lifecycle(obs_sst, obs_field, **kwargs)
    name = _clivar_name(prefix, variable, "TsRmse")
    return _result(rmse(model, obs, dim="lag"), name, model=model, obs=obs)


def _outside_box(data: xr.DataArray, box: dict) -> xr.DataArray:
    (lat0, lat1), (lon0, lon1) = box["lat"], box["lon"]
    inside = (data.lat > lat0) & (data.lat < lat1) & (data.lon > lon0) & (data.lon < lon1)
    return ~inside


def _pattern_statistics(model, obs, weighted) -> dict:
    valid = model.notnull() & obs.notnull()
    model, obs = model.where(valid), obs.where(valid)
    weights = np.cos(np.deg2rad(model.lat)) if weighted else xr.ones_like(model.lat)
    weights = weights.where(valid, 0)

    def mean(field):
        return field.weighted(weights).mean(["lat", "lon"])

    model_anomaly, obs_anomaly = model - mean(model), obs - mean(obs)
    model_std, obs_std = np.sqrt(mean(model_anomaly**2)), np.sqrt(mean(obs_anomaly**2))
    return {
        "value": np.sqrt(mean((model - obs) ** 2)),
        "correlation": mean(model_anomaly * obs_anomaly) / (model_std * obs_std),
        "std_ratio": model_std / obs_std,
    }


def enso_teleconnection(
    model_sst: xr.DataArray,
    obs_sst: xr.DataArray,
    model_field: xr.DataArray | None = None,
    obs_field: xr.DataArray | None = None,
    season: Literal["DJF", "JJA"] | str = "DJF",
    region: str = "global",
    index_region: str = "nino34",
    keep: xr.DataArray | None = None,
    weighted: bool = False,
    variable: str | None = "sst",
    kind: EnsoKind = "enso",
    threshold: float = 0.75,
) -> xr.Dataset:
    """
    Comparison of ENSO teleconnection maps
    (CLIVAR: EnsoSstMapDjf, EnsoSstMapJja, EnsoPrMapDjf, EnsoPrMapJja,
    EnsoSlpMapDjf, EnsoSlpMapJja, and the NinoSstMap, NinaSstMap,
    NinoPrMap, ... composites).

    The seasonal anomaly of the field is regressed without intercept onto the
    seasonal Niño 3.4 SST anomaly of the same season ("enso"), or averaged
    over the El Niño ("nino") or La Niña ("nina") events of each dataset,
    detected in that season, without smoothing. The maps are compared where both are valid and outside the equatorial
    Pacific, giving the RMSE (``value``), pattern correlation and ratio of
    spatial standard deviations, e.g. for a Taylor diagram.

    Parameters
    ----------
    model_sst, obs_sst
        Monthly SST used for the ENSO index.
    model_field, obs_field
        Monthly fields whose maps are compared, on the same grid. Default to
        the SST.
    season
        Season of the index and the maps.
    region
        Box of :data:`xenso.REGIONS` of the maps; "global" is 60°S-60°N.
    index_region
        Box of :data:`xenso.REGIONS` of the index.
    keep
        Boolean mask of the grid points to compare. By default the box
        15°S-15°N, 120°E-75°W is left out. CLIVAR leaves out the WOA09
        Pacific basin between 15°S and 15°N, which differs from the box near
        the coasts; pass such a mask to reproduce it.
    weighted
        Weight the statistics by cos(latitude). CLIVAR does not.
    variable
        Name used in ``clivar_name``, e.g. "pr" or "slp".
    kind
        "enso" for the regression, "nino" or "nina" for event composites.
    threshold
        Event threshold in standard deviations, for "nino" and "nina".
    """
    prefix = _prefix(kind)
    model_field = model_sst if model_field is None else model_field
    obs_field = obs_sst if obs_field is None else obs_field

    def regression_map(sst, field):
        return _enso_pattern(sst, field, index_region, region, season, None, kind, threshold)

    model, obs = _same_grid(regression_map(model_sst, model_field), regression_map(obs_sst, obs_field))
    if keep is None:
        keep = _outside_box(model, _TELECONNECTION_EXCLUDED)
    model, obs = model.where(keep), obs.where(keep)

    suffix = f"Map{str(season).capitalize()}" if kind == "enso" else "Map"
    name = _clivar_name(prefix, variable, suffix)
    return _result(**_pattern_statistics(model, obs, weighted), clivar_name=name, model=model, obs=obs)
