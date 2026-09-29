"""Region-based ENSO indices (Niño 1+2, 3, 3.4, 4, ONI, rONI) and box averages."""

from typing import Literal

import numpy as np
import xarray as xr

from .core import compute_anomaly
from .preprocessing import normalize_coords

_REGIONS: dict[str, dict] = {
    "12": {"lat": slice(-10, 0), "lon": slice(270, 280)},
    "3": {"lat": slice(-5, 5), "lon": slice(210, 270)},
    "34": {"lat": slice(-5, 5), "lon": slice(190, 240)},
    "4": {"lat": slice(-5, 5), "lon": slice(160, 210)},
}

_TROPICAL_DOMAIN = {"lat": slice(-20, 20)}

# Boxes used by the CLIVAR ENSO Metrics Package (EnsoCollectionsLib.ReferenceRegions)
REGIONS: dict[str, dict] = {
    "global": {"lat": slice(-60, 60), "lon": slice(0, 360)},
    "tropical_pacific": {"lat": slice(-30, 30), "lon": slice(120, 280)},
    "equatorial_pacific": {"lat": slice(-5, 5), "lon": slice(150, 270)},
    "equatorial_pacific_latext": {"lat": slice(-15, 15), "lon": slice(150, 270)},
    "equatorial_pacific_latext2": {"lat": slice(-15, 15), "lon": slice(120, 285)},
    "western_equatorial_pacific": {"lat": slice(-5, 5), "lon": slice(120, 205)},
    "eastern_equatorial_pacific": {"lat": slice(-5, 5), "lon": slice(205, 280)},
    "nino12": _REGIONS["12"],
    "nino3": _REGIONS["3"],
    "nino3_latext": {"lat": slice(-15, 15), "lon": slice(210, 270)},
    "nino34": _REGIONS["34"],
    "nino4": _REGIONS["4"],
}


def _box_mean(data: xr.DataArray, box: dict, weighted: bool) -> xr.DataArray:
    subset = normalize_coords(data).sel(**box)
    if weighted:
        subset = subset.weighted(np.cos(np.deg2rad(subset.lat)))
    return subset.mean(dim=["lat", "lon"])


def region_mean(
    data: xr.DataArray,
    region: str,
    weighted: bool = True,
) -> xr.DataArray:
    """
    Compute the spatial mean over one of the boxes in ``REGIONS``.

    Parameters
    ----------
    data
        DataArray with lat/lon dimensions.
    region
        Key of ``REGIONS``, e.g. "equatorial_pacific" or "nino3".
    weighted
        Weight by cos(latitude). Missing values are skipped and the weights
        renormalized.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}, expected one of {sorted(REGIONS)}")
    return _box_mean(data, REGIONS[region], weighted)


def nino_regions(
    data: xr.DataArray,
    region: Literal["12", "3", "34", "4"] = "34",
    weighted: bool = False,
) -> xr.DataArray:
    """
    Compute the spatial mean over the selected El Niño region.

    Parameters
    ----------
    data
        DataArray with lat/lon dimensions.
    region
        El Niño region: "12", "3", "34", or "4".
    weighted
        Weight by cos(latitude), as done in the CLIVAR ENSO Metrics Package.
    """
    return _box_mean(data, _REGIONS[region], weighted)


def oni(
    data: xr.DataArray,
    base_period: tuple[str, str] = ("1991-01-01", "2020-12-31"),
) -> xr.DataArray:
    """
    Compute the Oceanic Niño Index (ONI).

    ONI is the 3-month centered running mean of the Niño-3.4 SST anomaly.

    Parameters
    ----------
    data
        SST DataArray with time, lat, and lon dimensions.
    base_period
        Start and end dates used to compute the climatology.
    """
    nino34_anom = compute_anomaly(nino_regions(data, region="34"), base_period=base_period)
    return nino34_anom.rolling(time=3, center=True).mean().dropna("time")


def roni(
    data: xr.DataArray,
    base_period: tuple[str, str] = ("1991-01-01", "2020-12-31"),
) -> xr.DataArray:
    """
    Compute the Relative Oceanic Niño Index (rONI).

    rONI removes the tropical mean SST signal from the Niño-3.4 anomaly and
    rescales the result to preserve the original monthly variance, then applies
    a 3-month centered running mean.

    Parameters
    ----------
    data
        SST DataArray with time, lat, and lon dimensions.
    base_period
        Start and end dates used to compute the climatology.
    """
    nino34_anom = compute_anomaly(nino_regions(data, region="34"), base_period=base_period)
    trop_mean = normalize_coords(data).sel(**_TROPICAL_DOMAIN).mean(dim=["lat", "lon"])
    trop_anom = compute_anomaly(trop_mean, base_period=base_period)

    diff = nino34_anom - trop_anom

    # rescale month-by-month to preserve the original Niño-3.4 anomaly variance
    scaling = nino34_anom.groupby("time.month").std("time") / diff.groupby("time.month").std("time")
    scaled = (diff.groupby("time.month") * scaling).drop_vars("month")

    return scaled.rolling(time=3, center=True).mean().dropna("time")
