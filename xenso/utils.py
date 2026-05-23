"""Utility functions for preprocessing input data."""

import xarray as xr
from xarray.core.variable import MissingDimensionsError


def _check_dimensions(data: xr.DataArray, extra_dims: list[str] | None = None) -> None:
    """Raise if time or any extra dimension is missing from data."""
    if extra_dims is None:
        extra_dims = []
    data_dims = [dim.lower() for dim in data.dims]
    for dim in ["time"] + extra_dims:
        if dim not in data_dims:
            raise MissingDimensionsError(f"Could not find {dim} dimension")
