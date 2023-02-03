"""
Module containing the definitions and methods to compute
a variety of indices used to study ENSO
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import xarray as xr
from eofs.tools.standard import covariance_map
from eofs.xarray import Eof

from .core import compute_anomaly, compute_climatology, xconvolve


class ECindex:
    """
    Computes the E and C index according to Takahashi et al. 2011
    """

    def __init__(
        self,
        sst_data: xr.DataArray,
        isanomaly: bool = False,
        lat_range: Optional[Tuple[float, float]] = (-10, 10),
        long_range: Optional[Tuple[float, float]] = (110, 290),
        climatology: Optional[xr.DataArray] = None,
        base_period: Tuple[str, str] = ("1979-01-01", "2009-12-30"),
        corr_factor: Optional[List[int]] = None,
        smooth_kernel: List[int] = [1, 2, 1],
    ):
        self.sst_data = sst_data
        self.lat_range = lat_range
        self.long_range = long_range
        self.base_period = base_period
        if climatology is None and not isanomaly:
            climatology = compute_climatology(self.sst_data, base_period)
        self.climatology = climatology
        if not isanomaly:
            self.sst_data = compute_anomaly(self.sst_data, self.climatology)
        self._compute_pcs()
        self.smooth_kernel = smooth_kernel
        if corr_factor is None:
            self._auto_corr_factor()
        else:
            self.corr_factor = corr_factor

    def _compute_pcs(self) -> None:
        """
        Compute the principal components
        """
        _subset = self.sst_data.sortby(["lat", "lon"]).sel(
            lat=slice(*self.lat_range),  # type: ignore
            lon=slice(*self.long_range),  # type: ignore
        )
        if "month" in _subset.dims:
            _subset = _subset.drop("month")

        coslat = np.cos(np.deg2rad(_subset.lat.data))
        wgts = np.sqrt(coslat)[..., np.newaxis]

        self.solver = Eof(_subset.sel(time=slice(*self.base_period)), weights=wgts)
        self.anom_pcs = self.solver.projectField(_subset, neofs=2, eofscaling=1)
        self.anom_smooth_pcs = None

    def _corrected_pcs(self) -> xr.DataArray:
        """
        Return the pcs with the correction factor applied
        """
        return self.anom_pcs * self.corr_factor

    def _auto_corr_factor(self) -> None:
        """
        Automatically determine the correction factor by estimating
        the sign of known events for the E and C index.
        """
        _eofs = self.solver.eofs(neofs=2, eofscaling=1)
        _subset = dict(lat=slice(-5, 5), lon=slice(180, 200))
        new_corr_factor = np.zeros(2)
        new_corr_factor[0] = 1 if _eofs.sel(mode=0, **_subset).mean() > 0 else -1
        new_corr_factor[1] = 1 if _eofs.sel(mode=1, **_subset).mean() > 0 else -1
        self.corr_factor = new_corr_factor

    def _compute_index(self, smooth: bool = False) -> xr.Dataset:
        """
        Compute the E and C index
        """
        _pcs = self.pcs
        if smooth is True:
            _pcs = self.pcs_smooth
        pc1 = _pcs.sel(mode=0)
        pc2 = _pcs.sel(mode=1)
        eindex = (pc1 - pc2) / (2 ** (1 / 2))
        eindex.name = "E_index"
        cindex = (pc1 + pc2) / (2 ** (1 / 2))
        cindex.name = "C_index"
        return xr.merge([eindex, cindex])

    @property
    def corr_factor(self) -> xr.DataArray:
        """
        Return the correction factor applied to the first two pcs
        """
        return self._corr_factor

    @corr_factor.setter
    def corr_factor(self, corr_factor: List[int]) -> None:
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
    def smooth_kernel(self, smooth_kernel: List) -> None:
        """
        Set a new smooth kernel to be applied to the first two pcs
        """
        kernel = np.array(smooth_kernel)
        self._smooth_kernel = xr.DataArray(kernel / kernel.sum(), dims=["time"])

    @property
    def pcs(self) -> xr.DataArray:
        """
        Return the first two principal components used
        in the computation of the E and C index
        """
        return self._corrected_pcs()

    @property
    def pcs_smooth(self) -> xr.DataArray:
        """
        Return the first two principal components smoothed
        with the specified smooth_kernel
        """
        if self.anom_smooth_pcs is None:
            self.anom_smooth_pcs = xconvolve(
                self._corrected_pcs(),
                self._smooth_kernel,
                dim="time",
            )
        return self.anom_smooth_pcs

    @property
    def ecindex(self) -> xr.Dataset:
        """
        Return the first two principal components rotated,
        also known as the E and C index
        """
        return self._compute_index()

    @property
    def ecindex_smooth(self) -> xr.Dataset:
        """
        Return the first two principal components smoothed and
        rotated, also known as the E and C index
        """
        return self._compute_index(smooth=True)

    @property
    def eofs(self) -> xr.DataArray:
        """
        Returnt the first two corrected empirical orthogonal functions
        """
        return self.solver.eofs(neofs=2, eofscaling=1) * self.corr_factor

    @property
    def patterns(self) -> xr.Dataset:
        """
        Return the E and C patterns
        """
        _subsetEC = self.ecindex.sel(time=slice(*self.base_period))
        _indexdata = xr.concat([_subsetEC.E_index, _subsetEC.C_index], dim="mode").T
        reg_map = covariance_map(
            _indexdata.data,
            self.sst_data.sel(time=slice(*self.base_period)).data,
        ) / np.expand_dims(_indexdata.std(dim="time").data, axis=[1, 2])
        pattern = xr.Dataset(
            data_vars=dict(
                E_pattern=(["lat", "lon"], reg_map[0]),
                C_pattern=(["lat", "lon"], reg_map[1]),
            ),
            coords={"lat": self.sst_data.lat, "lon": self.sst_data.lon},
            attrs=dict(description="E and C regression patterns"),
        )

        return pattern

    @staticmethod
    def compute_alpha(pc1, pc2, return_fit=False):
        """
        Compute the alpha parameter used to measure the non-linearity of
        the E and C index
        """
        coefs = poly.polyfit(pc1, pc2, deg=2)
        if return_fit:
            xfit = np.arange(pc1.min(), pc1.max() + 0.1, 0.1)
            fit = poly.polyval(xfit, coefs)
            return coefs[-1], xfit, fit
        return coefs[-1]

    @staticmethod
    def plot_kiwi(pc1, pc2, ax: plt.Axes = None):
        """
        Makes the kiwi plot of the pc1 and pc2 for the mean of the
        DJF season
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        pc1 = pc1.sel(time=pc1.time.dt.month.isin([12, 1, 2]))
        pc1 = pc1.resample(time="QS-DEC").mean().dropna("time")

        pc2 = pc2.sel(time=pc2.time.dt.month.isin([12, 1, 2]))
        pc2 = pc2.resample(time="QS-DEC").mean().dropna("time")

        alpha, xfit, fit = ECindex.compute_alpha(pc1, pc2, return_fit=True)

        ax.axhline(0, color="k", linestyle="--", alpha=0.2)
        ax.axvline(0, color="k", linestyle="--", alpha=0.2)

        # draw a line 45 degrees
        x = np.linspace(-6, 6, 100)
        y = x
        ax.plot(x, y, color="k", alpha=0.5, lw=1)
        ax.plot(-x, y, color="k", alpha=0.5, lw=1)

        ax.scatter(
            pc1,
            pc2,
            s=8,
            marker="o",
            c="w",
            edgecolors="k",
            linewidths=0.5,
        )

        ax.plot(xfit, fit, c="r", label=f"$\\alpha=${alpha:.2f}")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.set_xlim(-4, 6)
        ax.set_ylim(-6, 4)
        ax.legend()


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
