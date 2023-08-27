"""
Utility module containing functions to preprocess the ingested data
"""
import xarray as xr
from xarray.core.variable import MissingDimensionsError


def _check_dimensions(data: xr.DataArray, extra_dims: list[str] = []) -> None:
    """
    Function to test if time dimension + any extra dimension is present
    in the DataArray
    """
    data_dims = [dim.lower() for dim in data.dims]
    for dim in ["time"] + extra_dims:
        if dim not in data_dims:
            raise MissingDimensionsError(f"Could not find {dim} dimension")
