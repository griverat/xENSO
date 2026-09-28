"""
ENSO event detection and event-centred sampling.

The definitions follow the CLIVAR ENSO Metrics Package
(https://github.com/CLIVAR-PRP/enso_metrics), which remains the reference
implementation. Differences from it are noted in each docstring.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

_MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_MONTH_INITIALS = "JFMAMJJASONDJFMAMJJASOND"


def _season_months(season: str | int) -> list[int]:
    """Translate 12, "DEC" or "NDJ" into a list of calendar months."""
    if isinstance(season, int) and 1 <= season <= 12:
        return [season]
    if isinstance(season, str):
        key = season.upper()
        if key in _MONTH_NAMES:
            return [_MONTH_NAMES.index(key) + 1]
        start = _MONTH_INITIALS.find(key)
        if 1 < len(key) <= 12 and start != -1:
            return [(start + ii) % 12 + 1 for ii in range(len(key))]
    raise ValueError(
        f"unknown season {season!r}: use a month number, a month name or initials such as 'NDJ'"
    )


def _event_years(years: Sequence[int] | xr.DataArray) -> np.ndarray:
    if isinstance(years, xr.DataArray):
        years = years["year"].values
    return np.asarray(years, dtype=int)


def seasonal_series(
    data: xr.DataArray,
    season: str | int = "DEC",
    dim: str = "time",
) -> xr.DataArray:
    """
    Average monthly data over one season per year.

    Seasons spanning two calendar years (e.g. "NDJ") are labelled with the
    year of their first month, so NDJ 1997 is Nov 1997 - Jan 1998. Incomplete
    seasons are dropped. Months are averaged without day-length weights.

    Parameters
    ----------
    data
        Monthly DataArray with a time dimension.
    season
        Month number (12), month name ("DEC") or consecutive month initials
        ("NDJ", "DJF", "MAM").
    dim
        Name of the time dimension.

    Returns
    -------
    DataArray with a ``year`` dimension replacing ``dim``.
    """
    months = _season_months(season)

    in_season = data[dim].dt.month.isin(months)
    subset = data.isel({dim: np.flatnonzero(in_season.values)})
    month = subset[dim].dt.month
    crosses_year = months != sorted(months)
    label = subset[dim].dt.year - ((month < months[0]) & crosses_year).astype(int)
    subset = subset.assign_coords(year=(dim, label.values))

    n_months = subset[dim].groupby("year").count()
    complete = n_months.year.values[(n_months == len(months)).values]
    return subset.groupby("year").mean(dim).sel(year=complete)


def detect_events(
    index: xr.DataArray,
    threshold: float = 0.75,
    season: str | int = "DEC",
    kind: Literal["nino", "nina"] = "nino",
    normalize: bool = True,
    dim: str = "time",
) -> xr.DataArray:
    """
    Detect El Niño or La Niña events from an index time series.

    The index is averaged over ``season`` for each year and its mean over all
    years removed. A year is an El Niño (La Niña) event when this value is
    above ``threshold`` (below ``-threshold``). The CLIVAR defaults are the
    Niño 3.4 SST anomaly, detrended and smoothed with a 5-month triangular
    window, in December, with a threshold of 0.75 standard deviations.

    Parameters
    ----------
    index
        One-dimensional monthly index, usually an SST anomaly.
    threshold
        Magnitude of the threshold, always positive. Its sign is set by ``kind``.
    season
        Month or season used for detection (see :func:`seasonal_series`).
    kind
        "nino" for warm events or "nina" for cold events.
    normalize
        If True, ``threshold`` is in units of the standard deviation
        (``ddof=0``) of the seasonal index.
    dim
        Name of the time dimension.

    Returns
    -------
    Seasonal index values of the detected events, with a ``year`` dimension.
    """
    if index.ndim != 1:
        raise ValueError(f"index must be one-dimensional, got dimensions {index.dims}")
    if kind not in ("nino", "nina"):
        raise ValueError(f"kind must be 'nino' or 'nina', got {kind!r}")

    seasonal = seasonal_series(index, season=season, dim=dim)
    seasonal = seasonal - seasonal.mean("year")
    if normalize:
        threshold = threshold * seasonal.std("year")

    is_event = seasonal > threshold if kind == "nino" else seasonal < -threshold
    return seasonal.isel(year=np.flatnonzero(is_event.values))


def event_windows(
    data: xr.DataArray,
    years: Sequence[int] | xr.DataArray,
    window: int | None = 6,
    month: int = 12,
    dim: str = "time",
) -> xr.DataArray:
    """
    Extract monthly data around each event.

    For an event in year ``Y`` the window spans January of ``Y + 1 - window // 2``
    to December of ``Y + window // 2``, as in the CLIVAR ENSO Metrics Package
    (6 years: Jan Y-2 to Dec Y+3). Months outside the record are NaN.

    Parameters
    ----------
    data
        Monthly DataArray with a time dimension.
    years
        Event years, or the output of :func:`detect_events`.
    window
        Even number of years in the window. If None, only ``month`` of each
        event year is returned.
    month
        Calendar month of the event peak, which is lag 0.
    dim
        Name of the time dimension.

    Returns
    -------
    DataArray with a ``year`` dimension and, unless ``window`` is None, a
    ``lag`` dimension in months relative to the event peak.
    """
    if window is not None and (window < 2 or window % 2):
        raise ValueError(f"window must be an even number of years, got {window}")
    years = _event_years(years)

    months_since_0 = (data[dim].dt.year * 12 + data[dim].dt.month - 1).values
    if len(np.unique(months_since_0)) != len(months_since_0):
        raise ValueError("data must have at most one time step per calendar month")

    peaks = years * 12 + month - 1
    if window is None:
        lags = np.array([0])
    else:
        lags = np.arange(window * 12) + (1 - window // 2) * 12 - (month - 1)
    targets = peaks[:, None] + lags[None, :]

    first = min(months_since_0.min(), targets.min())
    last = max(months_since_0.max(), targets.max())
    series = data.assign_coords({dim: months_since_0}).reindex({dim: np.arange(first, last + 1)})
    windows = series.sel({dim: xr.DataArray(targets, dims=["year", "lag"])}).drop_vars(dim)
    windows = windows.assign_coords(year=years, lag=lags)
    return windows.squeeze("lag", drop=True) if window is None else windows


def composite(
    data: xr.DataArray,
    years: Sequence[int] | xr.DataArray,
    window: int | None = 6,
    month: int = 12,
    dim: str = "time",
) -> xr.DataArray:
    """
    Average of :func:`event_windows` over events, skipping missing values.

    Parameters are the same as :func:`event_windows`.
    """
    return event_windows(data, years, window=window, month=month, dim=dim).mean("year")


def event_duration(
    lifecycle: xr.DataArray,
    threshold: float,
    kind: Literal["nino", "nina"] = "nino",
    dim: str = "lag",
) -> xr.DataArray:
    """
    Number of consecutive months around the peak beyond a threshold.

    Months are counted backwards from lag 0 (included) and forwards from lag 1
    while the lifecycle stays above ``threshold`` (below ``-threshold`` for
    La Niña). Missing values end the count. CLIVAR's EnsoDuration uses 0.25
    on the lagged regression onto the December Niño 3.4 index, and 0.5
    standard deviations for single events.

    Parameters
    ----------
    lifecycle
        Output of :func:`event_windows` or :func:`composite`, or any array with
        a ``lag`` dimension in months.
    threshold
        Magnitude of the threshold, always positive. Its sign is set by ``kind``.
    kind
        "nino" or "nina".
    dim
        Name of the lag dimension.
    """
    if kind not in ("nino", "nina"):
        raise ValueError(f"kind must be 'nino' or 'nina', got {kind!r}")
    beyond = lifecycle > threshold if kind == "nino" else lifecycle < -threshold
    beyond = beyond.astype(int)

    lags = lifecycle[dim].values
    before = beyond.isel({dim: np.flatnonzero(lags <= 0)[::-1]})
    after = beyond.isel({dim: np.flatnonzero(lags > 0)})
    duration = before.cumprod(dim).sum(dim) + after.cumprod(dim).sum(dim)
    return duration.rename("duration").assign_attrs(units="months")
