"""Input normalization to enforce consistent coordinate conventions."""

import xarray as xr


def normalize_coords(data: xr.DataArray) -> xr.DataArray:
    """
    Normalize spatial coordinates to standard conventions.

    - lat: ascending (S→N)
    - lon: 0–360 range, ascending (W→E)

    This ensures slice-based selections work correctly regardless of
    the source dataset's native ordering or lon convention.
    """
    if "lon" in data.dims:
        data = data.assign_coords(lon=data.lon % 360)
    dims_to_sort = [d for d in ["lat", "lon"] if d in data.dims]
    if dims_to_sort:
        data = data.sortby(dims_to_sort)
    return data
