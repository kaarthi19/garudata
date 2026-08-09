"""Lazy cutout view over the ERA5 weather store.

No materialized copies: a :class:`Cutout` is just an
``open_mfdataset`` over the monthly files that exist in the
``GEODATA_ROOT``-style store (``era5/<config>/<year>/<month:02d>.nc``),
subset to a bounding box.  geodata's cutout pipeline duplicated
gigabytes per cutout and its refactor cannot prepare ERA5 at all; this
view replaces it.

Two independent normalization layers:

* **Coordinates** (:func:`normalize_coords`, applied by ``Cutout.ds``):
  whatever the files carry (``latitude``/``longitude``, ``lat``/``lon``
  or ``x``/``y``; ``valid_time`` or ``time``), the returned dataset has
  dims ``x``/``y`` **and** ``lon``/``lat`` alias coordinates on those
  dims, so both geodata-refactor-style (``x``/``y``) and master-style
  (``lon``/``lat``) selection work.
* **Variables** (:func:`canonicalize`, applied by
  ``Cutout.canonical()``): the master-era raw-CDS -> canonical
  translation that the geodata refactor dropped — renames, derived
  variables (``influx_diffuse``, ``albedo``, ``wnd100m``, ``height``)
  and the J m-2 -> W m-2 conversion for radiation.  Idempotent.

ERA5 files carry latitude descending (north to south);
:func:`sel_bounds` handles either axis direction, so bbox selection
never silently returns an empty slice.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from .weather import _as_int_list, _months_list, era5_store_path, resolve_store_root

logger = logging.getLogger(__name__)

#: Standard gravity [m s-2] for geopotential -> height.
G0 = 9.80665

#: Raw-CDS short names consumed by :func:`canonicalize`.
_RAW_VARS = ("ssrd", "ssr", "fdir", "tisr", "u100", "v100", "z",
             "ro", "t2m", "sp", "stl4", "fsr", "d2m")

#: Simple raw -> canonical renames (master-era geodata names, including
#: the space in "soil temperature", kept for byte-compatibility with
#: master-built cutouts).
_CANONICAL_RENAMES = {
    "ro": "runoff",
    "t2m": "temperature",
    "sp": "pressure",
    "stl4": "soil temperature",
    "fsr": "roughness",
    "d2m": "dewpoint_temperature",
}


# ---------------------------------------------------------------------------
# Store inventory
# ---------------------------------------------------------------------------

def missing_files(
    years: int | slice | Iterable[int],
    months: int | slice | Iterable[int] | None = None,
    store_root: str | os.PathLike | None = None,
    config: str = "wind_solar_hourly",
) -> list[tuple[int, int, Path]]:
    """Report store gaps for a year/month request.

    Parameters
    ----------
    years : int, slice or iterable of int
        Years to check (slice is inclusive).
    months : int, slice, iterable of int, or None
        Months 1..12; ``None`` means all twelve.
    store_root : path-like, optional
        Store root; defaults to ``$GEODATA_ROOT``.
    config : str
        Store subdirectory under ``era5/``.

    Returns
    -------
    list of (year, month, path)
        One tuple per requested month whose file does not exist, with
        the path where :mod:`garudata.weather` would write it.  Empty
        list means the store is complete for the request.
    """
    root = resolve_store_root(store_root)
    gaps = [
        (y, m, p)
        for y in _as_int_list(years, "years")
        for m in _months_list(months)
        if not (p := era5_store_path(root, config, y, m)).exists()
    ]
    if gaps:
        logger.info(
            "store %s/era5/%s is missing %d of the requested files (first: %s)",
            root, config, len(gaps), gaps[0][2],
        )
    return gaps


# ---------------------------------------------------------------------------
# Coordinate normalization and bbox selection
# ---------------------------------------------------------------------------

def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Normalize spatial/time coordinate names; idempotent.

    Renames the longitude dim (``longitude``/``lon``/``x``) to ``x``
    and the latitude dim (``latitude``/``lat``/``y``) to ``y``, renames
    ``valid_time`` to ``time`` (new-format ERA5 files), then attaches
    ``lon`` and ``lat`` alias coordinates on the ``x``/``y`` dims so
    both naming conventions select correctly (aliases follow any later
    ``.sel``/``.isel`` automatically).

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Same data, dims ``x``/``y`` plus ``lon``/``lat`` aliases.
    """
    if ("valid_time" in ds.dims or "valid_time" in ds.coords) and "time" not in ds:
        ds = ds.rename({"valid_time": "time"})
    lon_dim = next((n for n in ("longitude", "lon", "x") if n in ds.dims), None)
    lat_dim = next((n for n in ("latitude", "lat", "y") if n in ds.dims), None)
    if lon_dim is None or lat_dim is None:
        raise ValueError(
            f"could not identify lon/lat dims among {tuple(ds.dims)}; "
            "expected longitude|lon|x and latitude|lat|y"
        )
    renames = {}
    if lon_dim != "x":
        renames[lon_dim] = "x"
    if lat_dim != "y":
        renames[lat_dim] = "y"
    if renames:
        ds = ds.rename(renames)
    if "lon" not in ds.coords:
        ds = ds.assign_coords(lon=("x", ds["x"].values))
    if "lat" not in ds.coords:
        ds = ds.assign_coords(lat=("y", ds["y"].values))
    return ds


def sel_bounds(ds: xr.Dataset, bounds_wsen: Sequence[float]) -> xr.Dataset:
    """Subset a coord-normalized dataset to canonical ``(W, S, E, N)`` bounds.

    Slice direction follows each axis's actual monotonic order, so the
    ERA5 descending-latitude convention is handled (a naive
    ``sel(y=slice(S, N))`` on descending latitude silently returns an
    empty axis).
    """
    if len(bounds_wsen) != 4:
        raise ValueError(f"bounds_wsen must have 4 elements (W, S, E, N), got {bounds_wsen!r}")
    w, s, e, n = (float(v) for v in bounds_wsen)
    if w > e or s > n:
        raise ValueError(f"bounds must be canonical (W, S, E, N), got {bounds_wsen!r}")
    xv = ds["x"].values
    yv = ds["y"].values
    xsl = slice(w, e) if (xv.size < 2 or xv[0] <= xv[-1]) else slice(e, w)
    ysl = slice(s, n) if (yv.size < 2 or yv[0] <= yv[-1]) else slice(n, s)
    return ds.sel(x=xsl, y=ysl)


# ---------------------------------------------------------------------------
# Raw -> canonical variable translation (master-era, dropped by refactor)
# ---------------------------------------------------------------------------

def is_canonical(ds: xr.Dataset) -> bool:
    """True if no raw-CDS variable (or ``valid_time``) remains in ``ds``."""
    if any(v in ds.data_vars for v in _RAW_VARS):
        return False
    return "valid_time" not in ds.dims and "valid_time" not in ds.coords


def canonicalize(ds: xr.Dataset) -> xr.Dataset:
    """Translate raw CDS variables to geodata's canonical schema.

    Mirrors master-era ``transform_wind_solar_dataset`` (the
    translation the geodata refactor dropped, leaving cutouts with raw
    ``ssrd/fdir/tisr/...`` that no converter understands):

    * ``valid_time`` -> ``time`` (new-format ERA5 files);
    * ``height = z / 9.80665`` (first time step; ``z`` dropped);
    * ``albedo = (ssrd - ssr) / ssrd`` — guarded division (``ssrd <= 0``
      -> NaN -> 0.0), clipped to [0, 1];
    * ``influx_diffuse = ssrd - fdir``;
    * renames ``fdir -> influx_direct``, ``tisr -> influx_toa``;
    * ``influx_direct``, ``influx_diffuse``, ``influx_toa`` clipped at
      0 and divided by 3600 (hourly J m-2 accumulation -> mean W m-2),
      ``units`` attr set to ``"W m**-2"``;
    * ``wnd100m = sqrt(u100^2 + v100^2)`` (``u100``/``v100`` dropped);
    * renames ``t2m -> temperature``, ``fsr -> roughness``,
      ``d2m -> dewpoint_temperature``, ``ro -> runoff``,
      ``sp -> pressure``, ``stl4 -> "soil temperature"``.

    Idempotent: every step is guarded on its raw inputs still being
    present, and the /3600 conversion is applied only to variables
    created in the same call — canonicalizing twice is a no-op.
    ``ssrd``/``ssr`` are dropped only after their derived products
    exist (safer than master's unconditional drop for partial variable
    sets).  Lazy (dask-backed) datasets stay lazy.

    Parameters
    ----------
    ds : xarray.Dataset
        Raw, partially canonical, or fully canonical dataset.

    Returns
    -------
    xarray.Dataset
        Canonical dataset (``influx_direct``, ``influx_diffuse``,
        ``influx_toa``, ``albedo``, ``temperature``, ``wnd100m``,
        ``roughness``, ...).
    """
    if is_canonical(ds):
        return ds

    # Shallow copy: in-place variable assignments below must never leak
    # into the caller's dataset (e.g. the cached ``Cutout.ds``).
    ds = ds.copy()

    if ("valid_time" in ds.dims or "valid_time" in ds.coords) and "time" not in ds:
        ds = ds.rename({"valid_time": "time"})

    # Geopotential -> geopotential height (time-invariant field).
    if "z" in ds.data_vars and "height" not in ds.data_vars:
        z = ds["z"]
        if "time" in z.dims:
            z = z.isel(time=0, drop=True)
        ds["height"] = (z / G0).assign_attrs(units="m", long_name="Geopotential height")
        ds = ds.drop_vars("z")

    # Derived radiation quantities from the raw names (before renames).
    to_convert: list[str] = []
    if "ssrd" in ds.data_vars and "ssr" in ds.data_vars and "albedo" not in ds.data_vars:
        albedo = (ds["ssrd"] - ds["ssr"]) / ds["ssrd"].where(ds["ssrd"] > 0)
        ds["albedo"] = (
            albedo.fillna(0.0)
            .clip(0.0, 1.0)
            .assign_attrs(units="(0 - 1)", long_name="Albedo")
        )
    if "ssrd" in ds.data_vars and "fdir" in ds.data_vars and "influx_diffuse" not in ds.data_vars:
        ds["influx_diffuse"] = (ds["ssrd"] - ds["fdir"]).assign_attrs(
            units="J m**-2",
            long_name="Surface diffuse solar radiation downwards",
        )
        to_convert.append("influx_diffuse")

    # Drop consumed inputs only once their derived products exist.
    drop = [
        v
        for v, product in (("ssrd", "influx_diffuse"), ("ssr", "albedo"))
        if v in ds.data_vars and product in ds.data_vars
    ]
    if drop:
        ds = ds.drop_vars(drop)

    if "fdir" in ds.data_vars and "influx_direct" not in ds.data_vars:
        ds = ds.rename({"fdir": "influx_direct"})
        to_convert.append("influx_direct")
    if "tisr" in ds.data_vars and "influx_toa" not in ds.data_vars:
        ds = ds.rename({"tisr": "influx_toa"})
        to_convert.append("influx_toa")

    # Hourly accumulations (J m-2) -> mean power (W m-2); only for the
    # variables created above, which keeps a second call a no-op.
    for name in to_convert:
        attrs = dict(ds[name].attrs)
        ds[name] = ds[name].clip(min=0.0) / 3600.0
        attrs["units"] = "W m**-2"
        ds[name].attrs = attrs

    if "u100" in ds.data_vars and "v100" in ds.data_vars and "wnd100m" not in ds.data_vars:
        ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
            units=ds["u100"].attrs.get("units", "m s**-1"),
            long_name="100 metre wind speed",
        )
        ds = ds.drop_vars(["u100", "v100"])

    renames = {
        k: v
        for k, v in _CANONICAL_RENAMES.items()
        if k in ds.data_vars and v not in ds.data_vars
    }
    if renames:
        ds = ds.rename(renames)
    return ds


# ---------------------------------------------------------------------------
# The lazy cutout
# ---------------------------------------------------------------------------

class Cutout:
    """Lazy view over the monthly ERA5 store — no materialized copies.

    Parameters
    ----------
    years : int, slice or iterable of int
        Years to include (slice is inclusive, geodata-style).
    months : int, slice, iterable of int, or None
        Months 1..12; ``None`` (default) means all twelve.
    bounds_wsen : sequence of float, optional
        Canonical ``(W, S, E, N)`` box the view is subset to;
        ``None`` keeps the files' full extent.
    store_root : path-like, optional
        Weather store root; defaults to ``$GEODATA_ROOT`` (resolved at
        construction, not import).
    config : str
        Store subdirectory under ``era5/``; default
        ``"wind_solar_hourly"``.
    open_kwargs : dict, optional
        Extra keyword arguments for ``xarray.open_mfdataset`` (e.g.
        ``{"chunks": {"time": 744}}``).

    Notes
    -----
    Construction touches no data.  ``.ds`` opens whatever requested
    files **exist** (missing months are tolerated — see
    :meth:`missing` / :func:`missing_files`) and raises
    ``FileNotFoundError`` only when none exist.  Multi-file opening
    uses dask via ``open_mfdataset``; a single file falls back to
    ``open_dataset`` so dask is not required for one-month views.

    Examples
    --------
    >>> co = Cutout(years=2018, bounds_wsen=(95, -11, 141, 6),
    ...             store_root="~/geodata_root")  # doctest: +SKIP
    >>> ds = co.canonical()  # doctest: +SKIP
    >>> ds["influx_direct"].attrs["units"]  # doctest: +SKIP
    'W m**-2'
    """

    def __init__(
        self,
        years: int | slice | Iterable[int],
        months: int | slice | Iterable[int] | None = None,
        bounds_wsen: Sequence[float] | None = None,
        *,
        store_root: str | os.PathLike | None = None,
        config: str = "wind_solar_hourly",
        open_kwargs: dict | None = None,
    ) -> None:
        self.years = _as_int_list(years, "years")
        self.months = _months_list(months)
        self.bounds_wsen = None if bounds_wsen is None else tuple(float(v) for v in bounds_wsen)
        if self.bounds_wsen is not None and len(self.bounds_wsen) != 4:
            raise ValueError(f"bounds_wsen must have 4 elements, got {bounds_wsen!r}")
        self.store_root = resolve_store_root(store_root)
        self.config = config
        self.open_kwargs = dict(open_kwargs or {})
        self._ds: xr.Dataset | None = None

    # -- store inventory ----------------------------------------------------

    @property
    def expected_files(self) -> list[Path]:
        """Store paths for every requested (year, month), existing or not."""
        return [
            era5_store_path(self.store_root, self.config, y, m)
            for y in self.years
            for m in self.months
        ]

    @property
    def files(self) -> list[Path]:
        """Sorted store files that exist for the request."""
        return sorted(p for p in self.expected_files if p.exists())

    def missing(self) -> list[tuple[int, int, Path]]:
        """Requested (year, month) pairs absent from the store."""
        return missing_files(self.years, self.months, self.store_root, self.config)

    # -- data access --------------------------------------------------------

    @property
    def ds(self) -> xr.Dataset:
        """The lazy dataset: existing files opened, coords normalized,
        subset to ``bounds_wsen``.  Opened once and cached; raw variable
        names are kept (use :meth:`canonical` for the canonical schema).
        """
        if self._ds is None:
            files = self.files
            if not files:
                raise FileNotFoundError(
                    f"no store files for years={self.years} months={self.months} under "
                    f"{self.store_root / 'era5' / self.config} — run "
                    "garudata.weather.download_era5_wind_solar first"
                )
            gaps = self.missing()
            if gaps:
                logger.warning(
                    "opening %d file(s); %d requested month(s) missing from the store "
                    "(first gap: %04d-%02d)",
                    len(files), len(gaps), gaps[0][0], gaps[0][1],
                )
            if len(files) == 1:
                ds = xr.open_dataset(files[0], **self.open_kwargs)
            else:
                kwargs = {"combine": "by_coords"}
                kwargs.update(self.open_kwargs)
                ds = xr.open_mfdataset([str(f) for f in files], **kwargs)
            ds = normalize_coords(ds)
            if self.bounds_wsen is not None:
                ds = sel_bounds(ds, self.bounds_wsen)
            self._ds = ds
        return self._ds

    def canonical(self) -> xr.Dataset:
        """Canonicalized view: :func:`canonicalize` applied to :attr:`ds`."""
        return canonicalize(self.ds)

    def close(self) -> None:
        """Close the underlying dataset (if opened) and drop the cache."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self) -> "Cutout":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        n_exp = len(self.expected_files)
        n_have = len(self.files)
        return (
            f"<garudata.Cutout config={self.config!r} years={self.years} "
            f"months={self.months} bounds_wsen={self.bounds_wsen} "
            f"files={n_have}/{n_exp} store={str(self.store_root)!r}>"
        )
