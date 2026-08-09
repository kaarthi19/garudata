"""Fixed-grid registry for reanalysis products.

Deterministic reconstruction of the ERA5 (0.25 deg) and MERRA-2
(0.625 x 0.5 deg) cell-center grids from a bounding box, plus
nearest-cell point mapping and cos-latitude cell areas.

Grid conventions
----------------
* ERA5 (CDS) netCDF files carry **longitude ascending** and **latitude
  descending** (north to south).  :func:`era5_grid` reproduces exactly
  that ordering so its output aligns index-for-index with a CDS
  download for the same ``area``.
* MERRA-2 (GES DISC) files carry both axes **ascending** (latitude
  south to north); :func:`merra2_grid` follows suit.
* Both products place cell centers on exact multiples of the grid step
  (ERA5: 0.25; MERRA-2: 0.625 lon / 0.5 lat — note -180 and -90 are
  themselves multiples).  A bbox therefore selects the grid points
  ``p`` with ``W <= p <= E`` / ``S <= p <= N`` (inclusive, with a small
  floating-point tolerance) — the same set CDS returns for
  ``area=[N, W, S, E]``.  To *cover* a bbox with whole cells rather
  than select interior centers, pad the bbox by half a step first.

Antimeridian-crossing boxes (W > E) are not supported.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

#: IUGG mean Earth radius [km].
EARTH_RADIUS_KM: float = 6371.0088

#: ERA5 single-levels grid step [deg] (both axes).
ERA5_STEP: float = 0.25

#: MERRA-2 grid steps [deg].
MERRA2_STEP_LON: float = 0.625
MERRA2_STEP_LAT: float = 0.5

_TOL = 1e-6  # tolerance in index space for inclusive bbox selection


def _validate_bbox(bbox_wsen: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate a canonical (W, S, E, N) bounding box."""
    if len(bbox_wsen) != 4:
        raise ValueError(f"bbox_wsen must have 4 elements (W, S, E, N), got {bbox_wsen!r}")
    w, s, e, n = (float(v) for v in bbox_wsen)
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValueError(f"longitudes out of [-180, 180]: W={w}, E={e}")
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValueError(f"latitudes out of [-90, 90]: S={s}, N={n}")
    if w > e:
        raise ValueError(
            f"W={w} > E={e}: antimeridian-crossing boxes are not supported "
            "(and check the (W, S, E, N) argument order)"
        )
    if s > n:
        raise ValueError(f"S={s} > N={n}: check the (W, S, E, N) argument order")
    return w, s, e, n


def _axis_points(lo: float, hi: float, step: float) -> NDArray[np.float64]:
    """All multiples of ``step`` in ``[lo, hi]`` (inclusive), ascending.

    Integer index arithmetic keeps the result exactly deterministic:
    ``k0 = ceil(lo/step)``, ``k1 = floor(hi/step)`` (with tolerance),
    values ``k*step`` for ``k`` in ``[k0, k1]``.
    """
    k0 = math.ceil(lo / step - _TOL)
    k1 = math.floor(hi / step + _TOL)
    if k1 < k0:
        logger.warning(
            "no grid points with step %s inside [%s, %s]; returning empty axis", step, lo, hi
        )
        return np.empty(0, dtype=np.float64)
    return np.arange(k0, k1 + 1, dtype=np.float64) * step


def era5_grid(
    bbox_wsen: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ERA5 0.25 deg cell centers inside a bounding box.

    Reproduces the grid a CDS request with ``area=[N, W, S, E]``
    returns: centers on exact multiples of 0.25 deg, longitudes
    ascending, **latitudes descending** (ERA5/CDS convention, north to
    south).  Selection is inclusive of grid points lying exactly on the
    bbox edges.

    Parameters
    ----------
    bbox_wsen : sequence of float
        Canonical bounding box ``(W, S, E, N)`` in degrees.

    Returns
    -------
    lons : numpy.ndarray
        1-D ascending longitudes [deg east].
    lats : numpy.ndarray
        1-D **descending** latitudes [deg north].

    Examples
    --------
    >>> lons, lats = era5_grid((95.0, -11.0, 141.0, 6.0))  # Indonesia
    >>> lons[0], lons[-1], lats[0], lats[-1]
    (95.0, 141.0, 6.0, -11.0)
    """
    w, s, e, n = _validate_bbox(bbox_wsen)
    lons = _axis_points(w, e, ERA5_STEP)
    lats = _axis_points(s, n, ERA5_STEP)[::-1].copy()  # descending, ERA5 convention
    return lons, lats


def merra2_grid(
    bbox_wsen: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """MERRA-2 0.625 x 0.5 deg cell centers inside a bounding box.

    Centers on exact multiples of 0.625 deg (longitude) and 0.5 deg
    (latitude); both axes **ascending** (MERRA-2 file convention,
    latitude south to north).

    Parameters
    ----------
    bbox_wsen : sequence of float
        Canonical bounding box ``(W, S, E, N)`` in degrees.

    Returns
    -------
    lons : numpy.ndarray
        1-D ascending longitudes [deg east].
    lats : numpy.ndarray
        1-D ascending latitudes [deg north].
    """
    w, s, e, n = _validate_bbox(bbox_wsen)
    lons = _axis_points(w, e, MERRA2_STEP_LON)
    lats = _axis_points(s, n, MERRA2_STEP_LAT)
    return lons, lats


def nearest_cell(
    lons: ArrayLike,
    lats: ArrayLike,
    pts_lon: ArrayLike,
    pts_lat: ArrayLike,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Indices of the grid cell nearest to each point.

    Works with either latitude ordering (ascending or descending) —
    nearest is by absolute coordinate difference per axis, which is
    exact for regular lon/lat grids.  Points outside the grid extent
    are clamped to the nearest edge cell (no error); check the returned
    distances yourself if that matters.

    Parameters
    ----------
    lons, lats : array_like
        1-D grid axes, e.g. from :func:`era5_grid`.
    pts_lon, pts_lat : array_like
        Point coordinates [deg]; scalars or 1-D arrays of equal length.

    Returns
    -------
    lat_idx, lon_idx : numpy.ndarray of intp
        Row/column indices into the grid, ordered for direct fancy
        indexing of ``(..., lat, lon)`` arrays:
        ``values[..., lat_idx, lon_idx]``.  Note the return order is
        (lat, lon) even though the argument order is (lons, lats).
    """
    glon = np.asarray(lons, dtype=np.float64)
    glat = np.asarray(lats, dtype=np.float64)
    if glon.ndim != 1 or glat.ndim != 1 or glon.size == 0 or glat.size == 0:
        raise ValueError("lons and lats must be non-empty 1-D arrays")
    plon = np.atleast_1d(np.asarray(pts_lon, dtype=np.float64))
    plat = np.atleast_1d(np.asarray(pts_lat, dtype=np.float64))
    if plon.shape != plat.shape:
        raise ValueError(f"pts_lon {plon.shape} and pts_lat {plat.shape} differ in shape")
    lon_idx = np.abs(glon[None, :] - plon[:, None]).argmin(axis=1)
    lat_idx = np.abs(glat[None, :] - plat[:, None]).argmin(axis=1)
    return lat_idx, lon_idx


def _uniform_step(values: NDArray[np.float64], name: str) -> float:
    """Infer the (uniform) spacing of a 1-D axis; raise if irregular."""
    if values.size < 2:
        raise ValueError(
            f"cannot infer {name} spacing from {values.size} point(s); "
            f"pass d{name} explicitly"
        )
    d = np.abs(np.diff(values))
    if float(d.max() - d.min()) > 1e-6:
        raise ValueError(f"{name} axis is not uniformly spaced (diffs {d.min()}..{d.max()})")
    return float(d.mean())


def cell_areas_km2(
    lons: ArrayLike,
    lats: ArrayLike,
    *,
    dlon: float | None = None,
    dlat: float | None = None,
) -> NDArray[np.float64]:
    """Cell areas [km^2] for a regular lon/lat grid (cos-lat weighting).

    Uses the exact spherical band formula
    ``A = R^2 * dlon_rad * (sin(lat + dlat/2) - sin(lat - dlat/2))``
    (which reduces to ``R^2 * dlon * dlat * cos(lat)`` to first order).
    Latitude edges are clipped to +/-90 deg so polar rows are handled
    correctly.  Works with either latitude ordering.

    Parameters
    ----------
    lons, lats : array_like
        1-D cell-center axes, e.g. from :func:`era5_grid`.
    dlon, dlat : float, optional
        Grid steps [deg]; inferred from the axes when omitted (requires
        at least two points per axis).

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(lats.size, lons.size)`` — row order
        matches ``lats`` as given, i.e. ``(y, x)`` indexing.
    """
    glon = np.asarray(lons, dtype=np.float64)
    glat = np.asarray(lats, dtype=np.float64)
    if glon.ndim != 1 or glat.ndim != 1:
        raise ValueError("lons and lats must be 1-D arrays")
    step_lon = float(dlon) if dlon is not None else _uniform_step(glon, "lon")
    step_lat = float(dlat) if dlat is not None else _uniform_step(glat, "lat")

    hi = np.minimum(glat + step_lat / 2.0, 90.0)
    lo = np.maximum(glat - step_lat / 2.0, -90.0)
    band = np.abs(np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)))  # (nlat,)
    row_area = EARTH_RADIUS_KM**2 * np.deg2rad(step_lon) * band
    return np.broadcast_to(row_area[:, None], (glat.size, glon.size)).copy()
