"""Region-based ENSO indices (Niño 1+2, 3, 3.4, 4, ONI, rONI)."""

from typing import Literal

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


def nino_regions(
    data: xr.DataArray,
    region: Literal["12", "3", "34", "4"] = "34",
) -> xr.DataArray:
    """
    Compute the spatial mean over the selected El Niño region.

    Parameters
    ----------
    data
        DataArray with lat/lon dimensions.
    region
        El Niño region: "12", "3", "34", or "4".
    """
    return normalize_coords(data).sel(**_REGIONS[region]).mean(dim=["lat", "lon"])


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
