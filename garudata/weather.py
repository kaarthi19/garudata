"""CDS ERA5 downloader with a GEODATA_ROOT-compatible store layout.

Replaces geodata's broken ERA5 download path.  Every request writes
exactly ``<store_root>/era5/<config>/<year>/<month:02d>.nc`` — the
cluster-compatible store contract — and fixes, by construction, the
confirmed geodata defects:

a. CDS ``area`` is built as ``[N, W, S, E]`` from canonical
   ``(W, S, E, N)`` bounds (geodata reversed the tuple, wrapping the
   antimeridian and downloading everything *except* the box).
b. Both response shapes are handled: the post-2024 CDS engine returns a
   **zip** for mixed instantaneous/accumulated variable sets and a
   **plain netCDF** for single-stream sets.  geodata only handled zip
   and silently saved nothing otherwise.  Detection is by magic bytes.
c. Downloads land in ``<final>.part`` and are moved to the final path
   with an atomic ``os.replace`` only after success — an interrupted
   run can never poison the store (geodata wrote the final path
   directly and then skipped the truncated file forever).
d. Existing files are skipped (resumable); pass
   ``check_integrity=True`` to also verify each existing file opens
   with xarray before trusting it.
e. :func:`download_era5_monthly_means` uses the real monthly-means
   product (``reanalysis-era5-single-levels-monthly-means`` with
   ``product_type='monthly_averaged_reanalysis'`` and **no** ``day``
   key) — geodata's "monthly" config requested hourly 00:00 UTC
   snapshots of the hourly product.
f. The default variable set is geodata's 13-variable
   ``wind_solar_hourly`` mix (:data:`DEFAULT_13`).

``cdsapi`` is imported lazily inside the download functions; importing
this module needs neither CDS credentials nor network access.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import xarray as xr

logger = logging.getLogger(__name__)

#: CDS product names.
ERA5_HOURLY_PRODUCT = "reanalysis-era5-single-levels"
ERA5_MONTHLY_PRODUCT = "reanalysis-era5-single-levels-monthly-means"

#: The 13-variable ERA5 ``wind_solar_hourly`` set (geodata
#: ``datasets/era5/hourly/wind_solar.py`` keys, order preserved).
DEFAULT_13: tuple[str, ...] = (
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "runoff",
    "soil_temperature_level_4",
    "surface_net_solar_radiation",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "toa_incident_solar_radiation",
    "total_sky_direct_solar_radiation_at_surface",
    "forecast_surface_roughness",
    "geopotential",
)

#: CDS long name -> netCDF short name, for reference and for
#: :func:`garudata.cutout.canonicalize` consumers.
CDS_TO_SHORT: dict[str, str] = {
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "runoff": "ro",
    "soil_temperature_level_4": "stl4",
    "surface_net_solar_radiation": "ssr",
    "surface_pressure": "sp",
    "surface_solar_radiation_downwards": "ssrd",
    "toa_incident_solar_radiation": "tisr",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "forecast_surface_roughness": "fsr",
    "geopotential": "z",
}


# ---------------------------------------------------------------------------
# Store layout helpers (the contract other modules import)
# ---------------------------------------------------------------------------

def resolve_store_root(store_root: str | os.PathLike | None = None) -> Path:
    """Resolve the weather store root.

    Parameters
    ----------
    store_root : path-like, optional
        Explicit root.  When ``None``, the ``GEODATA_ROOT`` environment
        variable is used (cluster convention).

    Returns
    -------
    pathlib.Path

    Raises
    ------
    ValueError
        If ``store_root`` is ``None`` and ``GEODATA_ROOT`` is unset.
    """
    if store_root is not None:
        return Path(store_root).expanduser()
    env = os.environ.get("GEODATA_ROOT", "")
    if not env:
        raise ValueError(
            "store_root not given and the GEODATA_ROOT environment variable is not set"
        )
    return Path(env).expanduser()


def era5_store_path(
    store_root: str | os.PathLike, config: str, year: int, month: int
) -> Path:
    """Canonical store path ``<store_root>/era5/<config>/<year>/<month:02d>.nc``."""
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return Path(store_root) / "era5" / config / str(int(year)) / f"{int(month):02d}.nc"


def cds_area(bounds_wsen: Sequence[float]) -> list[float]:
    """CDS ``area`` list ``[N, W, S, E]`` from canonical ``(W, S, E, N)`` bounds.

    This is the fix for geodata's reversed-bounds bug: canonical
    ``(W, S, E, N)`` maps to ``[b[3], b[0], b[1], b[2]]`` — never
    ``bounds[::-1]``, which swaps E/W and wraps the antimeridian.
    """
    if len(bounds_wsen) != 4:
        raise ValueError(f"bounds_wsen must have 4 elements (W, S, E, N), got {bounds_wsen!r}")
    w, s, e, n = (float(v) for v in bounds_wsen)
    if not (-180.0 <= w <= e <= 180.0):
        raise ValueError(
            f"expected -180 <= W <= E <= 180, got W={w}, E={e} — "
            "bounds must be canonical (W, S, E, N)"
        )
    if not (-90.0 <= s <= n <= 90.0):
        raise ValueError(
            f"expected -90 <= S <= N <= 90, got S={s}, N={n} — "
            "bounds must be canonical (W, S, E, N)"
        )
    return [n, w, s, e]


def _as_int_list(value: int | slice | Iterable[int] | None, name: str) -> list[int]:
    """Normalize an int / iterable / inclusive slice to a sorted int list."""
    if value is None:
        raise ValueError(f"{name} must not be None")
    if isinstance(value, slice):
        if value.start is None or value.stop is None:
            raise ValueError(f"{name} slice must have start and stop (inclusive)")
        return list(range(int(value.start), int(value.stop) + 1, int(value.step or 1)))
    if isinstance(value, int):
        return [value]
    out = sorted({int(v) for v in value})
    if not out:
        raise ValueError(f"{name} is empty")
    return out


def _months_list(months: int | slice | Iterable[int] | None) -> list[int]:
    ms = list(range(1, 13)) if months is None else _as_int_list(months, "months")
    bad = [m for m in ms if not 1 <= m <= 12]
    if bad:
        raise ValueError(f"months out of range 1..12: {bad}")
    return ms


# ---------------------------------------------------------------------------
# Download core
# ---------------------------------------------------------------------------

def _openable(path: Path) -> bool:
    """True if ``path`` opens as a netCDF dataset (cheap integrity check)."""
    try:
        with xr.open_dataset(path):
            return True
    except Exception as exc:  # noqa: BLE001 — any failure means "not usable"
        logger.warning("integrity check failed for %s: %s", path, exc)
        return False


def _finalize(part: Path, final: Path) -> None:
    """Turn a completed ``.part`` download into the final store file.

    Inspects magic bytes: a ``PK`` zip wrapper (post-2024 CDS behavior
    for mixed variable streams) is unwrapped — single member moved,
    multiple members merged with ``open_mfdataset`` — while a plain
    netCDF is moved as-is.  The final path only ever appears via
    ``os.replace`` (atomic within the same filesystem; all temporaries
    live in ``final.parent``).
    """
    with open(part, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] != b"PK":
        os.replace(part, final)
        return

    with tempfile.TemporaryDirectory(dir=final.parent) as tmpdir:
        try:
            with zipfile.ZipFile(part) as zf:
                zf.extractall(tmpdir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"{part} is a truncated/corrupt zip: {exc}") from exc
        ncs = sorted(glob.glob(os.path.join(tmpdir, "**", "*.nc"), recursive=True))
        if not ncs:
            raise RuntimeError(f"no .nc member inside CDS zip {part}")
        if len(ncs) == 1:
            os.replace(ncs[0], final)
        else:
            merged = os.path.join(tmpdir, "__merged__.nc4")
            with xr.open_mfdataset(ncs, combine="by_coords") as ds:
                ds.to_netcdf(merged)
            os.replace(merged, final)
    part.unlink(missing_ok=True)


def _download_one(
    client,
    product: str,
    request: dict,
    final: Path,
    retries: int,
) -> None:
    """Retrieve one request to ``final`` atomically (via ``.part``)."""
    final.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(final) + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            logger.info("CDS retrieve %s -> %s (attempt %d)", product, final, attempt)
            client.retrieve(product, request, str(part))
            _finalize(part, final)
            return
        except Exception as exc:  # noqa: BLE001 — retried, then re-raised
            last_exc = exc
            part.unlink(missing_ok=True)
            logger.warning(
                "download attempt %d/%d for %s failed: %s", attempt, retries, final, exc
            )
    part.unlink(missing_ok=True)
    raise RuntimeError(f"download failed after {retries} attempts: {final}") from last_exc


def _run_downloads(
    product: str,
    build_request,
    years: int | slice | Iterable[int],
    months: int | slice | Iterable[int] | None,
    store_root: str | os.PathLike | None,
    config: str,
    skip_existing: bool,
    check_integrity: bool,
    retries: int,
) -> list[Path]:
    """Shared driver: worklist, skip/integrity logic, lazy cdsapi client."""
    root = resolve_store_root(store_root)
    ys = _as_int_list(years, "years")
    ms = _months_list(months)

    targets = [(y, m, era5_store_path(root, config, y, m)) for y in ys for m in ms]
    todo: list[tuple[int, int, Path]] = []
    for y, m, path in targets:
        if path.exists() and skip_existing:
            if check_integrity and not _openable(path):
                logger.warning("existing %s failed integrity check; re-downloading", path)
                todo.append((y, m, path))
            else:
                logger.debug("skip existing %s", path)
        else:
            todo.append((y, m, path))

    if not todo:
        logger.info("store complete: %d file(s), nothing to download", len(targets))
        return [p for _, _, p in targets]

    import cdsapi  # lazy: credentials/network only needed when there is work

    client = cdsapi.Client()
    for y, m, path in todo:
        _download_one(client, product, build_request(y, m), path, retries)
    logger.info("downloaded %d file(s) into %s", len(todo), root / "era5" / config)
    return [p for _, _, p in targets]


# ---------------------------------------------------------------------------
# Public download functions
# ---------------------------------------------------------------------------

def download_era5_wind_solar(
    years: int | slice | Iterable[int],
    months: int | slice | Iterable[int] | None,
    bounds_wsen: Sequence[float] | None,
    store_root: str | os.PathLike | None = None,
    config: str = "wind_solar_hourly",
    variables: Sequence[str] = DEFAULT_13,
    *,
    skip_existing: bool = True,
    check_integrity: bool = False,
    retries: int = 3,
) -> list[Path]:
    """Download hourly ERA5 single-levels months into the weather store.

    Writes exactly ``<store_root>/era5/<config>/<year>/<month:02d>.nc``
    (one file per calendar month, full days, 24 hourly steps), the
    layout :class:`garudata.cutout.Cutout` and the lab cluster expect.

    Parameters
    ----------
    years : int, slice or iterable of int
        Years to fetch.  A slice is inclusive (``slice(2018, 2019)`` ==
        ``[2018, 2019]``, matching geodata's convention).
    months : int, slice, iterable of int, or None
        Months (1..12); ``None`` means all twelve.
    bounds_wsen : sequence of float or None
        Canonical ``(W, S, E, N)`` box; converted internally to CDS
        ``area=[N, W, S, E]`` by :func:`cds_area`.  ``None`` downloads
        the global grid (large!).
    store_root : path-like, optional
        Store root; defaults to ``$GEODATA_ROOT``.
    config : str
        Store subdirectory under ``era5/``; default
        ``"wind_solar_hourly"``.
    variables : sequence of str
        CDS variable long names; default the 13-variable geodata
        ``wind_solar_hourly`` set (:data:`DEFAULT_13`).
    skip_existing : bool
        Skip months already in the store (resumability).
    check_integrity : bool
        With ``skip_existing``, also verify existing files open with
        xarray and re-download the ones that do not.
    retries : int
        Attempts per file before raising.

    Returns
    -------
    list of pathlib.Path
        Final store paths for **all** requested months (downloaded or
        pre-existing), sorted by (year, month).

    Notes
    -----
    Interrupted runs never leave a partial file at a final path: data
    is written to ``<final>.nc.part`` and moved with ``os.replace``
    only after the response is unwrapped (zip or plain netCDF) — see
    :func:`_finalize`.  Requires ``~/.cdsapirc`` credentials at call
    time; ``cdsapi`` is imported lazily.
    """
    variables = list(variables)
    area = None if bounds_wsen is None else cds_area(bounds_wsen)

    def build_request(year: int, month: int) -> dict:
        req = {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
        }
        if area is not None:
            req["area"] = area
        return req

    return _run_downloads(
        ERA5_HOURLY_PRODUCT,
        build_request,
        years,
        months,
        store_root,
        config,
        skip_existing,
        check_integrity,
        retries,
    )


def download_era5_monthly_means(
    years: int | slice | Iterable[int],
    months: int | slice | Iterable[int] | None,
    bounds_wsen: Sequence[float] | None,
    store_root: str | os.PathLike | None = None,
    config: str = "wind_solar_monthly",
    variables: Sequence[str] = DEFAULT_13,
    *,
    skip_existing: bool = True,
    check_integrity: bool = False,
    retries: int = 3,
) -> list[Path]:
    """Download true ERA5 monthly means into the weather store.

    Uses product ``reanalysis-era5-single-levels-monthly-means`` with
    ``product_type='monthly_averaged_reanalysis'`` and **no** ``day``
    key (``time='00:00'`` is the product's nominal timestamp) — the
    correct request geodata's "monthly" config never made (it fetched
    hourly 00:00 UTC snapshots of the hourly product instead).

    Store layout, arguments and atomic/resumable behavior are identical
    to :func:`download_era5_wind_solar`; files land at
    ``<store_root>/era5/<config>/<year>/<month:02d>.nc``.

    See Also
    --------
    download_era5_wind_solar : parameter documentation.
    """
    variables = list(variables)
    area = None if bounds_wsen is None else cds_area(bounds_wsen)

    def build_request(year: int, month: int) -> dict:
        req = {
            "product_type": "monthly_averaged_reanalysis",
            "data_format": "netcdf",
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "time": "00:00",  # nominal timestamp; no 'day' key for monthly means
        }
        if area is not None:
            req["area"] = area
        return req

    return _run_downloads(
        ERA5_MONTHLY_PRODUCT,
        build_request,
        years,
        months,
        store_root,
        config,
        skip_existing,
        check_integrity,
        retries,
    )
