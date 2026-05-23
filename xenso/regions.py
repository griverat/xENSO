"""Region-based ENSO indices (Niño 1+2, 3, 3.4, 4)."""

from typing import Literal

import xarray as xr

_REGIONS: dict[str, dict] = {
    "12": {"lat": slice(-10, 0), "lon": slice(270, 280)},
    "3": {"lat": slice(-5, 5), "lon": slice(210, 270)},
    "34": {"lat": slice(-5, 5), "lon": slice(190, 240)},
    "4": {"lat": slice(-5, 5), "lon": slice(160, 210)},
}


def nino_regions(
    data: xr.DataArray,
    region: Literal["12", "3", "34", "4"] = "34",
) -> xr.DataArray:
    """
    Compute the mean over the selected El Niño region.

    Parameters
    ----------
    data
        DataArray with lat/lon dimensions.
    region
        El Niño region: "12", "3", "34", or "4".
    """
    return data.sel(**_REGIONS[region]).mean(dim=["lat", "lon"])
