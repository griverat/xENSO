"""
ENSO diagnostics of a single dataset.

Each function computes one diagnostic of the CLIVAR ENSO Metrics Package
(https://github.com/CLIVAR-PRP/enso_metrics), named in its docstring and in the
``clivar_name`` attribute of the result. Inputs are gridded monthly fields;
the result is a Dataset with a ``value`` variable and, where CLIVAR defines
one, an ``error`` variable. Pass the model and observed results to
:func:`xenso.compare` to obtain the metric.

As in CLIVAR, fields are converted to anomalies relative to the climatology
of the whole record, linearly detrended and, for the event-based
diagnostics, smoothed with a 5-month triangular window. Here the box average
is taken first, while CLIVAR averages last; the two only differ where
missing values change within the box. CLIVAR also regrids every field to a
1° grid, which is left to the user. This is not the official
implementation: use the original package for results that must be
comparable with published CLIVAR ENSO metrics.
"""

from typing import Literal

import numpy as np
import xarray as xr

from .core import compute_anomaly, detrend, smooth
from .events import composite, detect_events, event_duration, event_windows, seasonal_series
from .preprocessing import normalize_coords
from .regions import REGIONS, region_mean
from .stats import linregress, split_regression

EnsoKind = Literal["enso", "nino", "nina"]
_PREFIX = {"enso": "Enso", "nino": "Nino", "nina": "Nina"}

# W m-2 during one month to temperature change of a 50 m slab of sea water
_SECONDS_PER_MONTH = 60 * 60 * 24 * 30.42
_SEAWATER_HEAT_CAPACITY = 4000 * 1024  # J K-1 m-3


def _anomalies(
    data: xr.DataArray,
    region: str | None = None,
    smoothing: int | None = None,
) -> xr.DataArray:
    """Box average, full-record anomalies, linear detrend and optional smoothing."""
    if region is not None:
        data = region_mean(data, region)
    anomalies = compute_anomaly(data, base_period=(None, None)).drop_vars("month")
    anomalies = detrend(anomalies)
    if smoothing:
        anomalies = smooth(anomalies, window=smoothing)
    return anomalies


def _prefix(kind: str) -> str:
    """CLIVAR name prefix for regression-based ("enso") or composite ("nino", "nina") metrics."""
    if kind not in _PREFIX:
        raise ValueError(f"kind must be one of {sorted(_PREFIX)}, got {kind!r}")
    return _PREFIX[kind]


def _lifecycle(
    sst: xr.DataArray,
    field: xr.DataArray | None = None,
    region: str = "nino34",
    field_region: str | None = None,
    month: int = 12,
    window: int = 6,
    smoothing: int | None = 5,
    kind: EnsoKind = "enso",
    threshold: float = 0.75,
) -> xr.DataArray:
    """
    ENSO life cycle of ``field`` (default: the SST index itself) along a ``lag`` dimension.

    "enso": regression, without intercept, onto the index in ``month`` of each year.
    "nino"/"nina": composite over the events detected in the index.
    """
    _prefix(kind)
    index = _anomalies(sst, region, smoothing)
    series = index if field is None else _anomalies(field, field_region or region, smoothing)
    if kind == "enso":
        peak = seasonal_series(index, month)
        peak = peak - peak.mean("year")
        windows = event_windows(series, peak.year, window=window, month=month)
        return linregress(windows, peak, dim="year", fit_intercept=False).slope
    events = detect_events(index, threshold=threshold, season=month, kind=kind)
    return composite(series, events, window=window, month=month)


def _zonal_section(data: xr.DataArray, region: str) -> xr.DataArray:
    """Cos(latitude)-weighted meridional mean over a box of ``REGIONS``."""
    box = normalize_coords(data).sel(**REGIONS[region])
    return box.weighted(np.cos(np.deg2rad(box.lat))).mean("lat")


def _n_years(data: xr.DataArray) -> int:
    return round(data.sizes["time"] / 12)


def _result(value, clivar_name: str, units: str | None = None, error=None, **extra) -> xr.Dataset:
    variables = {"value": value, **extra}
    if error is not None:
        variables["error"] = error
    result = xr.Dataset(variables, attrs={"clivar_name": clivar_name})
    for variable in result.variables.values():
        variable.attrs = {}
    if units is not None:
        result["value"].attrs["units"] = units
    return result


def _ratio_units(numerator: xr.DataArray, denominator: xr.DataArray) -> str | None:
    if "units" in numerator.attrs and "units" in denominator.attrs:
        return f"{numerator.attrs['units']} / ({denominator.attrs['units']})"
    return None


def enso_amplitude(sst: xr.DataArray, region: str = "nino34") -> xr.Dataset:
    """
    Standard deviation of the SST anomaly in a box (CLIVAR: EnsoAmpl).

    The error is the standard deviation divided by the square root of the
    number of years.

    Parameters
    ----------
    sst
        Monthly SST with time, lat and lon dimensions.
    region
        Key of :data:`xenso.REGIONS`.
    """
    index = _anomalies(sst, region)
    value = index.std("time")
    return _result(value, "EnsoAmpl", sst.attrs.get("units"), error=value / np.sqrt(_n_years(sst)))


def enso_seasonality(sst: xr.DataArray, region: str = "nino34") -> xr.Dataset:
    """
    Ratio of the NDJ to the MAM standard deviation of the SST anomaly
    (CLIVAR: EnsoSeasonality).

    Parameters
    ----------
    sst
        Monthly SST with time, lat and lon dimensions.
    region
        Key of :data:`xenso.REGIONS`.
    """
    index = _anomalies(sst, region)
    ndj = seasonal_series(index, "NDJ").std("year")
    mam = seasonal_series(index, "MAM").std("year")
    n_years = _n_years(sst)
    ndj_err, mam_err = ndj / np.sqrt(n_years - 1), mam / np.sqrt(n_years)
    error = (mam * ndj_err + ndj * mam_err) / mam**2
    return _result(ndj / mam, "EnsoSeasonality", "1", error=error, ndj_std=ndj, mam_std=mam)


def enso_skewness(sst: xr.DataArray, region: str = "nino34") -> xr.Dataset:
    """
    Skewness of the SST anomaly in a box (CLIVAR: EnsoSstSkew).

    The skewness is the biased estimator of ``scipy.stats.skew``. The error
    follows CLIVAR: the skewness divided by the square root of the number of
    years.

    Parameters
    ----------
    sst
        Monthly SST with time, lat and lon dimensions.
    region
        Key of :data:`xenso.REGIONS`.
    """
    index = _anomalies(sst, region)
    deviation = index - index.mean("time")
    value = (deviation**3).mean("time") / (deviation**2).mean("time") ** 1.5
    return _result(value, "EnsoSstSkew", "1", error=value / np.sqrt(_n_years(sst)))


def enso_duration(
    sst: xr.DataArray,
    region: str = "nino34",
    threshold: float = 0.25,
    month: int = 12,
    window: int = 6,
    smoothing: int | None = 5,
) -> xr.Dataset:
    """
    Duration of the ENSO life cycle (CLIVAR: EnsoDuration).

    The monthly SST anomaly around each year is regressed, without intercept,
    onto the anomaly in ``month`` of that year. The duration is the number
    of consecutive months around ``month`` during which the regression
    exceeds ``threshold``.

    Parameters
    ----------
    sst
        Monthly SST with time, lat and lon dimensions.
    region
        Key of :data:`xenso.REGIONS`.
    threshold
        Threshold on the regression coefficient.
    month
        Calendar month of the ENSO peak.
    window
        Even number of years in the life cycle (see :func:`xenso.event_windows`).
    smoothing
        Length of the triangular running mean applied first, or None.

    Returns
    -------
    Dataset with the duration in months as ``value`` and the regression
    life cycle as ``lifecycle``.
    """
    lifecycle = _lifecycle(sst, region=region, month=month, window=window, smoothing=smoothing)
    value = event_duration(lifecycle, threshold)
    return _result(value, "EnsoDuration", "months", lifecycle=lifecycle)


def enso_event_duration(
    sst: xr.DataArray,
    kind: Literal["nino", "nina"] = "nino",
    region: str = "nino34",
    threshold: float = 0.75,
    duration_threshold: float = 0.5,
    month: int = 12,
    window: int = 6,
    smoothing: int | None = 5,
) -> xr.Dataset:
    """
    Mean duration of El Niño or La Niña events (CLIVAR: NinoSstDur, NinaSstDur).

    Events are detected with :func:`xenso.detect_events` in ``month``. The
    duration of each event is the number of consecutive months around the
    peak during which the index stays beyond ``duration_threshold`` standard
    deviations of the monthly index (see :func:`xenso.event_duration`).

    Parameters
    ----------
    sst
        Monthly SST with time, lat and lon dimensions.
    kind
        "nino" or "nina".
    region
        Key of :data:`xenso.REGIONS`.
    threshold
        Event threshold in standard deviations of the index in ``month``.
    duration_threshold
        Threshold of the duration in standard deviations of the monthly index.
    month
        Calendar month of the event peak.
    window
        Even number of years around each event (see :func:`xenso.event_windows`).
    smoothing
        Length of the triangular running mean applied first, or None.

    Returns
    -------
    Dataset with the mean duration in months as ``value``, its standard error
    as ``error`` and the duration of each event as ``durations``.
    """
    if kind not in ("nino", "nina"):
        raise ValueError(f"kind must be 'nino' or 'nina', got {kind!r}")
    index = _anomalies(sst, region, smoothing)
    events = detect_events(index, threshold=threshold, season=month, kind=kind)
    windows = event_windows(index, events, window=window, month=month)
    durations = event_duration(windows, duration_threshold * index.std("time"), kind=kind)
    error = durations.std("year") / np.sqrt(durations.sizes["year"])
    return _result(
        durations.mean("year"), f"{_prefix(kind)}SstDur", "months", error=error, durations=durations
    )


def enso_diversity(
    sst: xr.DataArray,
    region: str = "equatorial_pacific",
    event_region: str = "nino34",
    threshold: float = 0.75,
    season: str | int = "DEC",
    smoothing: int | None = 5,
    lon_smoothing: int = 5,
    kind: EnsoKind = "enso",
    east_of: float = 220.0,
) -> xr.Dataset:
    """
    Spread of the longitude of the peak SST anomaly among ENSO events
    (CLIVAR: EnsoSstDiversity, NinoSstDiversity, NinaSstDiversity).

    Events are detected with :func:`xenso.detect_events` (normalized
    threshold) in ``event_region``. For each event the equatorial SST anomaly
    profile in ``season`` is smoothed in longitude and the longitude of its
    maximum (El Niño) or minimum (La Niña) is taken. The diversity is the
    interquartile range of these longitudes over all events, or over El Niño
    or La Niña events only. Also returned is the percentage of events peaking
    east of ``east_of`` (CLIVAR: NinoSstDiv, NinaSstDiv). The peak longitude
    of a single event can jump between two nearly equal maxima, so the
    diversity changes in steps with small changes of the input.

    Parameters
    ----------
    sst
        Monthly SST with time, lat and lon dimensions. CLIVAR regrids it to
        1° first, which ``lon_smoothing`` assumes.
    region
        Box of :data:`xenso.REGIONS` over which the zonal profile is taken.
    event_region
        Box of :data:`xenso.REGIONS` used to detect events.
    threshold
        Event threshold in standard deviations.
    season
        Month or season used to detect events and take the profile.
    smoothing
        Length of the triangular running mean in time, or None.
    lon_smoothing
        Number of longitude points of the triangular smoothing of each profile.
    kind
        "enso" for all events, "nino" or "nina" for one kind.
    east_of
        Longitude, in °E, separating eastern Pacific events (CLIVAR: 140°W).

    Returns
    -------
    Dataset with the interquartile range as ``value``, the median absolute
    deviation as ``mad``, the percentage of eastern Pacific events as
    ``eastern_fraction`` and the longitude of each event as ``peak_lon``.
    """
    prefix = _prefix(kind)
    index = _anomalies(sst, event_region, smoothing)
    nino = detect_events(index, threshold=threshold, season=season, kind="nino")
    nina = detect_events(index, threshold=threshold, season=season, kind="nina")
    if kind == "nino":
        nina = nina.isel(year=slice(0, 0))
    elif kind == "nina":
        nino = nino.isel(year=slice(0, 0))

    profiles = seasonal_series(_anomalies(_zonal_section(sst, region), smoothing=smoothing), season)
    profiles = smooth(profiles - profiles.mean("year"), window=lon_smoothing, dim="lon")
    peak_lon = xr.concat(
        [
            profiles.sel(year=nino.year)
            .idxmax("lon")
            .assign_coords(kind=("year", ["nino"] * nino.size)),
            profiles.sel(year=nina.year)
            .idxmin("lon")
            .assign_coords(kind=("year", ["nina"] * nina.size)),
        ],
        dim="year",
    ).sortby("year")

    quartiles = peak_lon.quantile([0.25, 0.75], dim="year")
    value = quartiles.sel(quantile=0.75, drop=True) - quartiles.sel(quantile=0.25, drop=True)
    mad = abs(peak_lon - peak_lon.median("year")).median("year")
    eastern_fraction = 100 * (peak_lon > east_of).mean("year")
    return _result(
        value,
        f"{prefix}SstDiversity",
        "degrees",
        mad=mad,
        eastern_fraction=eastern_fraction,
        peak_lon=peak_lon,
    )


def _feedback(response, forcing, response_region, forcing_region, clivar_name) -> xr.Dataset:
    response, forcing = xr.align(
        _anomalies(response, response_region), _anomalies(forcing, forcing_region)
    )
    fit = split_regression(response, forcing)
    return _result(
        fit.slope.sel(branch="all", drop=True),
        clivar_name,
        _ratio_units(response, forcing),
        error=fit.stderr.sel(branch="all", drop=True),
        **fit,
    )


def bjerknes_feedback(
    sst: xr.DataArray,
    taux: xr.DataArray,
    sst_region: str = "nino3",
    taux_region: str = "nino4",
) -> xr.Dataset:
    """
    Regression of the zonal wind stress anomaly onto the SST anomaly
    (CLIVAR: EnsoFbSstTaux).

    CLIVAR expresses zonal wind stress in 1e-3 N m-2; convert ``taux`` first to
    obtain the same numbers.

    Returns
    -------
    Dataset with the slope over all months as ``value`` and its standard error
    as ``error``, plus the output of :func:`xenso.split_regression`.
    """
    return _feedback(taux, sst, taux_region, sst_region, "EnsoFbSstTaux")


def heat_flux_feedback(
    sst: xr.DataArray,
    thf: xr.DataArray,
    sst_region: str = "nino3",
    thf_region: str = "nino3",
) -> xr.Dataset:
    """
    Regression of the net surface heat flux anomaly onto the SST anomaly
    (CLIVAR: EnsoFbSstThf).

    ``thf`` is the sum of the net shortwave, net longwave, latent and
    sensible heat fluxes, positive into the ocean, so the feedback is
    negative when the fluxes damp SST anomalies.

    The result has the same variables as :func:`bjerknes_feedback`.
    """
    return _feedback(thf, sst, thf_region, sst_region, "EnsoFbSstThf")


def thermocline_feedback(
    sst: xr.DataArray,
    ssh: xr.DataArray,
    sst_region: str = "nino3",
    ssh_region: str = "nino3",
) -> xr.Dataset:
    """
    Regression of the SST anomaly onto the sea surface height anomaly
    (CLIVAR: EnsoFbSshSst).

    CLIVAR expresses sea surface height in cm; convert ``ssh`` first to obtain
    the same numbers.

    The result has the same variables as :func:`bjerknes_feedback`.
    """
    return _feedback(sst, ssh, sst_region, ssh_region, "EnsoFbSshSst")


def wind_ssh_feedback(
    taux: xr.DataArray,
    ssh: xr.DataArray,
    taux_region: str = "nino4",
    ssh_region: str = "nino3",
) -> xr.Dataset:
    """
    Regression of the sea surface height anomaly onto the zonal wind stress
    anomaly (CLIVAR: EnsoFbTauxSsh).

    CLIVAR expresses sea surface height in cm and zonal wind stress in
    1e-3 N m-2; convert the inputs first to obtain the same numbers.

    The result has the same variables as :func:`bjerknes_feedback`.
    """
    return _feedback(ssh, taux, ssh_region, taux_region, "EnsoFbTauxSsh")


def ocean_driven_sst_change(
    sst: xr.DataArray,
    thf: xr.DataArray,
    region: str = "nino3",
    event_region: str = "nino34",
    threshold: float = 0.75,
    season: str | int = "DEC",
    smoothing: int | None = 5,
    mixed_layer_depth: float = 50.0,
    min_change: float = 0.1,
) -> xr.Dataset:
    """
    Fraction of the June-December SST change of ENSO events not explained by
    surface heat fluxes (CLIVAR: EnsodSstOce, after Bayr et al. 2018).

    For each El Niño and La Niña event, the SST change since June and the
    change a ``mixed_layer_depth`` slab would get from the accumulated heat
    flux anomaly since July are divided by the June-December SST change.
    Events with a June-December change smaller than ``min_change`` are
    skipped. The ocean-driven part is their difference, averaged over events.

    Parameters
    ----------
    sst
        Monthly SST in °C with time, lat and lon dimensions.
    thf
        Monthly net surface heat flux in W m-2, positive into the ocean.
    region
        Box of :data:`xenso.REGIONS` where the SST budget is computed.
    event_region
        Box of :data:`xenso.REGIONS` used to detect events.
    threshold
        Event threshold in standard deviations.
    season
        Month or season used to detect events.
    smoothing
        Length of the triangular running mean in time, or None.
    mixed_layer_depth
        Depth of the slab ocean in m.
    min_change
        Smallest June-December SST change, in °C, for an event to be used.

    Returns
    -------
    Dataset with the ocean-driven change in December as ``value``, and the
    event-averaged ``dsst``, ``dsst_thf`` and ``dsst_oce`` from June to
    December along a ``month`` dimension.
    """
    index = _anomalies(sst, event_region, smoothing)
    events = np.concatenate(
        [
            detect_events(index, threshold=threshold, season=season, kind=kind).year.values
            for kind in ("nino", "nina")
        ]
    )

    sst_box, thf_box = xr.align(_anomalies(sst, region, smoothing), _anomalies(thf, region, smoothing))
    june_to_december = {"lag": slice(-6, 0)}
    sst_events = event_windows(sst_box, events, window=2).sel(june_to_december)
    thf_events = event_windows(thf_box, events, window=2).sel(june_to_december)

    to_temperature = _SECONDS_PER_MONTH / (_SEAWATER_HEAT_CAPACITY * mixed_layer_depth)
    dsst = sst_events - sst_events.isel(lag=0)
    dsst_thf = to_temperature * thf_events.where(thf_events.lag > -6, 0).cumsum("lag")
    total = dsst.isel(lag=-1)
    total = total.where(abs(total) >= min_change)

    budget = xr.Dataset(
        {"dsst": dsst / total, "dsst_thf": dsst_thf / total, "dsst_oce": (dsst - dsst_thf) / total}
    ).mean("year")
    budget = budget.assign_coords(month=("lag", budget.lag.values + 12)).swap_dims(lag="month")
    budget = budget.drop_vars("lag")
    return _result(budget.dsst_oce.sel(month=12, drop=True), "EnsodSstOce", "1", **budget)
