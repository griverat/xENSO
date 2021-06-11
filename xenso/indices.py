from typing import Optional

import numpy as np
import xarray as xr
from core import compute_anomaly, compute_climatology, xconvolve
from eofs.xarray import Eof


class ECindex:
    """
    Computes the E and C index according to Takahashi
    """

    def __init__(
        self,
        sst_data: xr.DataArray,
        isanomaly: bool = False,
        climatology: Optional[xr.DataArray] = None,
        base_period: tuple[str, str] = ("1979-01-01", "2009-12-30"),
    ):
        self.sst_data = sst_data
        if climatology is None:
            climatology = compute_climatology(self.sst_data, base_period)
        self.climatology = climatology
        if isanomaly:
            self.sst_data = compute_anomaly(self.sst_data, self.climatology)

    def _compute_pcs(self, corr_factor=[1, -1], smooth_kernel=[1, 2, 1]):
        coslat = np.cos(np.deg2rad(self.climatology.lat.data))
        wgts = np.sqrt(coslat)[..., np.newaxis]

        if corr_factor is None:
            corr_factor = [1, 1]
        corr_factor = xr.DataArray(np.array(corr_factor), coords=[("mode", [0, 1])])
        self.solver = Eof(self.climatology, weights=wgts)
        clim_std = self.solver.eigenvalues(neigs=2) ** (1 / 2)
        self.anom_pcs = (
            self.solver.projectField(
                self.sst_anom.sortby("lat").sel(lat=slice(-10, 10)),
                neofs=2,
            )
            * corr_factor
            / clim_std
        )
        kernel = np.array(smooth_kernel)
        kernel = xr.DataArray(kernel / kernel.sum(), dims=["time"])
        self.anom_smooth_pcs = xconvolve(self.anom_pcs, kernel, dim="time")

    def _compute_index(self, smooth=False):
        if smooth is True:
            pc1 = self.anom_smooth_pcs.sel(mode=0)
            pc2 = self.anom_smooth_pcs.sel(mode=1)
        else:
            pc1 = self.anom_pcs.sel(mode=0)
            pc2 = self.anom_pcs.sel(mode=1)
        cindex = (pc1 + pc2) / (2 ** (1 / 2))
        eindex = (pc1 - pc2) / (2 ** (1 / 2))
        return eindex, cindex

    @property
    def get_pcs(self):
        return self.anom_pcs

    @property
    def get_pcs_smooth(self):
        return self.anom_smooth_pcs

    @property
    def get_index(self):
        return self._compute_index()

    @property
    def get_smoothed_index(self):
        return self._compute_index(smooth=True)
