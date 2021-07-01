"""
Module containing the definitions and methods to compute
a variety of indices used to study ENSO
"""

from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from eofs.xarray import Eof

from .core import compute_anomaly, compute_climatology, xconvolve


class ECindex:
    """
    Computes the E and C index according to Takahashi
    """

    def __init__(
        self,
        sst_data: xr.DataArray,
        isanomaly: bool = False,
        climatology: Optional[xr.DataArray] = None,
        base_period: Tuple[str, str] = ("1979-01-01", "2009-12-30"),
        corr_factor: List = [1, 1],
        smooth_kernel: List = [1, 2, 1],
    ):
        self.sst_data = sst_data
        self.base_period = base_period
        if climatology is None:
            climatology = compute_climatology(self.sst_data, base_period)
        self.climatology = climatology
        if not isanomaly:
            self.sst_data = compute_anomaly(self.sst_data, self.climatology)
        self._corr_factor = xr.DataArray(
            np.array(corr_factor),
            coords=[("mode", [0, 1])],
        )
        self._smooth_kernel = xr.DataArray(
            np.array(smooth_kernel) / np.array(smooth_kernel).sum(),
            dims=["time"],
        )
        self._compute_pcs()

    def _compute_pcs(self) -> None:
        """
        Compute the principal components
        """
        _subset = self.sst_data.sortby("lat").sel(
            time=slice(*self.base_period),
            lat=slice(-10, 10),
        )

        coslat = np.cos(np.deg2rad(_subset.lat.data))
        wgts = np.sqrt(coslat)[..., np.newaxis]

        self.solver = Eof(_subset, weights=wgts)
        clim_std = self.solver.eigenvalues(neigs=2) ** (1 / 2)
        self.anom_pcs = (
            self.solver.projectField(
                _subset,
                neofs=2,
            )
            / clim_std
        )
        self.anom_smooth_pcs = None

    def _corrected_pcs(self):
        """
        Return the pcs with the correction factor applied
        """
        return self.anom_pcs * self._corr_factor

    def _compute_index(self, smooth: bool = False) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute the E and C index
        """
        _pcs = self._corrected_pcs()
        if smooth is True:
            _pcs = xconvolve(self.anom_pcs, self._smooth_kernel, dim="time")
        pc1 = _pcs.sel(mode=0)
        pc2 = _pcs.sel(mode=1)
        cindex = (pc1 + pc2) / (2 ** (1 / 2))
        eindex = (pc1 - pc2) / (2 ** (1 / 2))
        return eindex, cindex

    @property
    def corr_factor(self) -> xr.DataArray:
        """
        Return the correction factor applied to the first two pcs
        """
        return self._corr_factor

    @corr_factor.setter
    def corr_factor(self, corr_factor: List):
        """
        Set a new correction factor to be applied to the first two pcs
        """
        self._corr_factor = xr.DataArray(
            np.array(corr_factor),
            coords=[("mode", [0, 1])],
        )

    @property
    def smooth_kernel(self) -> xr.DataArray:
        """
        Return the smooth kernel used in the first two pcs
        """
        return self._smooth_kernel

    @smooth_kernel.setter
    def smooth_kernel(self, smooth_kernel: List):
        """
        Set a new smooth kernel to be applied to the first two pcs
        """
        kernel = np.array(smooth_kernel)
        self._smoothkernel = xr.DataArray(kernel / kernel.sum(), dims=["time"])

    @property
    def get_pcs(self) -> xr.DataArray:
        """
        Return the first two principal components used
        in the computation of the E and C index
        """
        return self.anom_pcs

    @property
    def get_pcs_smooth(self) -> xr.DataArray:
        """
        Return the first two principal components smoothed
        with the specified smooth_kernel
        """
        if self.anom_smooth_pcs is None:
            self.anom_smooth_pcs = xconvolve(
                self.anom_pcs,
                self._smooth_kernel,
                dim="time",
            )
        return self.anom_smooth_pcs

    @property
    def get_index(self) -> xr.DataArray:
        """
        Return the first two principal components rotated,
        also known as the E and C index
        """
        return self._compute_index()

    @property
    def get_smoothed_index(self) -> xr.DataArray:
        """
        Return the first two principal components smoothed and
        rotated, also known as the E and C index
        """
        return self._compute_index(smooth=True)


def enzones(data: xr.DataArray, zone: str = "34") -> xr.DataArray:
    """
    Computes the mean from the selected El Niño zone, also
    know as El Niño Index for each of the zones.
    """
    zones = {
        "12": {"lat": slice(-10, 0), "lon": slice(270, 280)},
        "3": {"lat": slice(-5, 5), "lon": slice(210, 270)},
        "34": {"lat": slice(-5, 5), "lon": slice(190, 240)},
        "4": {"lat": slice(-5, 5), "lon": slice(160, 210)},
    }
    return data.sel(**zones[zone]).mean(dim=["lat", "lon"])
