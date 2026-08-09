"""Windowed, native-CRS, class-resolved land-cover area engine.

Port of the validated ``china_landuse`` pipeline (notebook
``geodata_kag.ipynb``, CLCD section — the "79-second national pass"),
generalised for the GARUDA interface contract.

Algorithm (from the china_landuse handoff notes):

1. windowed read of the raster in its **native CRS** (no whole-raster
   reprojection, no full-band loads — bounded memory even for a 22 GB
   national 30 m raster);
2. block-reduce each window to ``block x block``-pixel samples: per-block,
   per-class pixel counts plus one representative block-center coordinate;
3. transform block centers to EPSG:4326 with a single ``pyproj.Transformer``;
4. assign each block center to a target-grid cell (nearest cell center);
5. accumulate per-(cell, class) areas with ``np.bincount``.

Pixel areas come from the native resolution: constant ``|a*e| / 1e6`` km2
for projected (assumed equal-area / low-distortion metric) rasters, or a
per-sample cos-latitude correction for geographic rasters.

Exclusion samplers are callables ``(lons, lats) -> bool`` evaluated at the
block centers (EPSG:4326); ``True`` means the sample **passes** (is kept).
Nodata never passes: the provided :func:`terrain_sampler` uses masked reads
and treats nodata / out-of-extent points as failing, fixing the China
notebook's "NaN slope -> not excluded" leak.

Notes
-----
* Window edges are trimmed to whole blocks (``h -= h % block``), exactly as
  in the validated pipeline; the sub-block remainder rows/columns of each
  window are dropped. This is part of the validated behaviour (golden
  number: 9,493,438 km2 classified for CLCD 2019).
* ``fraction`` is area divided by the *geometric* cell area (cos-lat,
  111.32 km/deg convention shared with the rest of garudata). Each block
  is assigned wholly to its nearest cell, so per-cell fractions carry
  discretization noise of order ``block_extent / cell_extent`` (and can
  exceed 1 for fully covered cells) while totals are conserved; with the
  validated 300 m blocks on the 0.25-degree grid this is ~1 %.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.windows
import xarray as xr
from pyproj import Transformer

logger = logging.getLogger(__name__)

__all__ = [
    "class_area_by_cell",
    "suitable_area_by_cell",
    "terrain_sampler",
    "polygon_sampler",
]

#: km per degree of latitude (and of longitude at the equator); the same
#: convention the captive notebook and the China pipeline use.
KM_PER_DEG = 111.32

#: An exclusion sampler: (lons, lats) in EPSG:4326 -> bool array,
#: True where the sample PASSES (is kept).
ExclusionSampler = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Largest raster class code for which a dense lookup table is built;
# beyond this a sorted searchsorted mapping is used instead.
_LUT_MAX_CODE = 1_000_000


def _grid_axis(values: Sequence[float], cell_size_deg: float | None, name: str) -> tuple[np.ndarray, float]:
    """Validate a 1-D grid-center axis; return (ascending centers, spacing)."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array of cell centers")
    if arr.size >= 2 and arr[1] < arr[0]:
        arr = arr[::-1]
    arr = np.ascontiguousarray(arr)
    if arr.size >= 2:
        d = np.diff(arr)
        if np.any(d <= 0):
            raise ValueError(f"{name} centers must be strictly monotonic")
        if not np.allclose(d, d[0], rtol=1e-4, atol=1e-9):
            raise ValueError(f"{name} centers are not uniformly spaced")
        return arr, float(d[0])
    if cell_size_deg is None:
        raise ValueError(
            f"{name} has a single center; pass cell_size_deg to define the cell size"
        )
    return arr, float(cell_size_deg)


def _code_mapper(
    class_map: Mapping[int, str], labels: list[str]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return f(int array) -> int32 class-index array (-1 = unmapped)."""
    codes = np.array([int(k) for k in class_map], dtype=np.int64)
    idx = np.array([labels.index(v) for v in class_map.values()], dtype=np.int32)
    if codes.size == 0:
        raise ValueError("class_map is empty")
    if np.unique(codes).size != codes.size:
        raise ValueError("class_map has duplicate raster codes")

    if codes.min() >= 0 and codes.max() <= _LUT_MAX_CODE:
        mx = int(codes.max())
        lut = np.full(mx + 1, -1, dtype=np.int32)
        lut[codes] = idx

        def mapper(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(np.int64, copy=False)
            out = lut[np.clip(a, 0, mx)]
            out[(a < 0) | (a > mx)] = -1
            return out

    else:
        order = np.argsort(codes)
        sc, si = codes[order], idx[order]

        def mapper(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(np.int64, copy=False)
            pos = np.clip(np.searchsorted(sc, a), 0, sc.size - 1)
            out = np.where(sc[pos] == a, si[pos], np.int32(-1)).astype(np.int32)
            return out

    return mapper


def class_area_by_cell(
    raster_path: str,
    class_map: Mapping[int, str],
    grid_lons: Sequence[float],
    grid_lats: Sequence[float],
    *,
    exclusion_samplers: Sequence[ExclusionSampler] = (),
    window: int = 8192,
    block: int = 10,
    cell_size_deg: float | None = None,
) -> xr.Dataset:
    """Per-class land area (km2) and fraction on a regular lon/lat target grid.

    Streams the raster in ``window x window`` native-CRS windows, reduces
    each to ``block x block``-pixel samples, transforms block centers to
    EPSG:4326 and accumulates per-(cell, class) areas with ``np.bincount``.
    Memory use is bounded by one window plus the (n_class, n_cell)
    accumulators, so a 22 GB national raster is fine.

    Parameters
    ----------
    raster_path : str
        Path to a single-band categorical raster (any CRS; not rotated).
    class_map : mapping of int to str
        Raster code -> class label. Codes absent from the mapping
        (typically background/nodata) are ignored. Several codes may map
        to one label (crosswalk collapse, e.g. CLCD -> RESDC).
    grid_lons, grid_lats : sequence of float
        Target-grid **cell centers** in degrees, uniformly spaced,
        monotonic (either direction; output is ascending). E.g. the fixed
        ERA5 0.25-degree centers.
    exclusion_samplers : sequence of callable, optional
        Each maps ``(lons, lats)`` (EPSG:4326 block centers) to a boolean
        array, ``True`` = sample passes. Samples must pass *all* samplers
        to count toward ``area``. See :func:`terrain_sampler` and
        :func:`polygon_sampler`.
    window : int, optional
        Read-window edge in pixels (default 8192).
    block : int, optional
        Block-reduce factor in pixels (default 10; 30 m pixels -> 300 m
        samples).
    cell_size_deg : float, optional
        Cell size override; only required when a grid axis has a single
        center (spacing cannot be inferred).

    Returns
    -------
    xarray.Dataset
        Dims ``(land_class, lat, lon)`` with:

        - ``area`` — km2 of each class per cell, after exclusions;
        - ``fraction`` — ``area`` / geometric cell area (cos-lat);
        - ``cell_area`` — geometric cell area, km2, dims ``(lat, lon)``;
        - ``area_unmasked`` — km2 before exclusions (only present when
          ``exclusion_samplers`` is non-empty).

    Raises
    ------
    ValueError
        On empty/duplicated class maps, non-uniform grids, rotated rasters
        or invalid window/block sizes.
    """
    if block < 1 or window < block:
        raise ValueError(f"need window >= block >= 1 (got window={window}, block={block})")

    lons_c, dlon = _grid_axis(grid_lons, cell_size_deg, "grid_lons")
    lats_c, dlat = _grid_axis(grid_lats, cell_size_deg, "grid_lats")
    nx, ny = lons_c.size, lats_c.size
    lon0, lat0 = float(lons_c[0]), float(lats_c[0])
    ncells = nx * ny

    labels = list(dict.fromkeys(class_map.values()))
    ncls = len(labels)
    mapper = _code_mapper(class_map, labels)
    samplers = tuple(exclusion_samplers)

    acc_all = np.zeros((ncls, ncells), dtype=np.float64)
    acc_keep = np.zeros((ncls, ncells), dtype=np.float64) if samplers else acc_all

    with rasterio.open(raster_path) as src:
        tr = src.transform
        if tr.b != 0 or tr.d != 0:
            raise ValueError("rotated rasters are not supported")
        if src.nodata is not None and int(src.nodata) in {int(k) for k in class_map}:
            logger.warning(
                "raster nodata value %s is also a mapped class in class_map; "
                "nodata pixels will be counted as that class", src.nodata
            )

        geographic = bool(src.crs is not None and src.crs.is_geographic)
        if geographic:
            px_base = abs(tr.a) * KM_PER_DEG * abs(tr.e) * KM_PER_DEG  # x cos(lat)
            logger.info("pixel area: geographic CRS, cos-lat corrected (%.3g km2 at equator)", px_base)
        else:
            px_area_const = abs(tr.a * tr.e) / 1e6
            logger.info(
                "pixel area: projected CRS %s, constant %.6g km2/pixel "
                "(assumed equal-area / low-distortion metric)",
                src.crs, px_area_const,
            )

        transformer = None
        if src.crs is not None and src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        elif src.crs is None:
            logger.warning("raster has no CRS; coordinates assumed EPSG:4326")

        nwy = (src.height + window - 1) // window
        nwx = (src.width + window - 1) // window
        for wy in range(nwy):
            for wx in range(nwx):
                row0, col0 = wy * window, wx * window
                h = min(window, src.height - row0)
                w = min(window, src.width - col0)
                h -= h % block
                w -= w % block
                if h <= 0 or w <= 0:
                    continue
                arr = src.read(1, window=rasterio.windows.Window(col0, row0, w, h))
                cls = mapper(arr)
                if not (cls >= 0).any():
                    continue

                nby, nbx = h // block, w // block
                blocks = (
                    cls.reshape(nby, block, nbx, block)
                    .transpose(0, 2, 1, 3)
                    .reshape(nby * nbx, block * block)
                )
                rr, cc = np.meshgrid(
                    row0 + (np.arange(nby) + 0.5) * block,
                    col0 + (np.arange(nbx) + 0.5) * block,
                    indexing="ij",
                )
                xs, ys = rasterio.transform.xy(tr, rr.ravel(), cc.ravel())
                xs, ys = np.asarray(xs), np.asarray(ys)
                if transformer is not None:
                    lons, lats = transformer.transform(xs, ys)
                else:
                    lons, lats = xs, ys

                ix = np.round((lons - lon0) / dlon).astype(np.int64)
                iy = np.round((lats - lat0) / dlat).astype(np.int64)
                inbox = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                keep = inbox
                for s in samplers:
                    keep = keep & np.asarray(s(lons, lats), dtype=bool)
                cell = iy * nx + ix  # only valid where inbox

                if geographic:
                    parea = px_base * np.cos(np.deg2rad(ys))  # native y = latitude
                else:
                    parea = px_area_const

                for ci in range(ncls):
                    cnt = (blocks == ci).sum(axis=1)
                    nz = cnt > 0
                    if not nz.any():
                        continue
                    warea = cnt * parea
                    sel = inbox & nz
                    if sel.any():
                        acc_all[ci] += np.bincount(
                            cell[sel], weights=warea[sel], minlength=ncells
                        )
                    if samplers:
                        sel = keep & nz
                        if sel.any():
                            acc_keep[ci] += np.bincount(
                                cell[sel], weights=warea[sel], minlength=ncells
                            )
            logger.info("window row %d/%d done", wy + 1, nwy)

    aw = dlon * KM_PER_DEG * np.cos(np.deg2rad(lats_c))
    cell_area = np.outer(aw * (dlat * KM_PER_DEG), np.ones(nx))

    area_keep = acc_keep.reshape(ncls, ny, nx)
    ds = xr.Dataset(
        {
            "area": (("land_class", "lat", "lon"), area_keep),
            "fraction": (("land_class", "lat", "lon"), area_keep / cell_area),
            "cell_area": (("lat", "lon"), cell_area),
        },
        coords={"land_class": labels, "lat": lats_c, "lon": lons_c},
        attrs={
            "source_raster": str(raster_path),
            "window_px": window,
            "block_px": block,
            "pixel_area_mode": "cos-lat (geographic CRS)" if geographic else "native metric",
            "n_exclusion_samplers": len(samplers),
            "units": "km2 per cell per class; fraction of geometric cell area",
            "method": (
                "windowed native-CRS read -> block-center samples -> pyproj to "
                "EPSG:4326 -> nearest-cell bincount (china_landuse validated engine)"
            ),
        },
    )
    if samplers:
        ds["area_unmasked"] = (("land_class", "lat", "lon"), acc_all.reshape(ncls, ny, nx))
    logger.info(
        "class_area_by_cell: %.0f km2 total (%s)%s",
        float(acc_keep.sum()),
        ", ".join(labels),
        f"; {float(acc_all.sum()):.0f} km2 before exclusions" if samplers else "",
    )
    return ds


def suitable_area_by_cell(
    raster_path: str,
    suitable_values: Iterable[int],
    grid_lons: Sequence[float],
    grid_lats: Sequence[float],
    *,
    exclusion_samplers: Sequence[ExclusionSampler] = (),
    window: int = 8192,
    block: int = 10,
    cell_size_deg: float | None = None,
) -> xr.Dataset:
    """Binary-suitability convenience over :func:`class_area_by_cell`.

    All ``suitable_values`` collapse into one "suitable" class; a single
    windowed pass produces per-cell suitable ``area`` (km2), ``fraction``
    and ``cell_area`` (dims ``(lat, lon)``; the class dim is dropped).

    Parameters
    ----------
    raster_path : str
        Categorical raster path.
    suitable_values : iterable of int
        Raster codes considered suitable.
    grid_lons, grid_lats, exclusion_samplers, window, block, cell_size_deg
        As in :func:`class_area_by_cell`.

    Returns
    -------
    xarray.Dataset
        Same variables as :func:`class_area_by_cell` without the
        ``land_class`` dimension.
    """
    class_map = {int(v): "suitable" for v in suitable_values}
    if not class_map:
        raise ValueError("suitable_values is empty")
    ds = class_area_by_cell(
        raster_path,
        class_map,
        grid_lons,
        grid_lats,
        exclusion_samplers=exclusion_samplers,
        window=window,
        block=block,
        cell_size_deg=cell_size_deg,
    )
    return ds.sel(land_class="suitable", drop=True)


def terrain_sampler(
    raster_path: str,
    max_value: float,
    *,
    min_value: float | None = None,
    band: int = 1,
) -> ExclusionSampler:
    """Exclusion-sampler factory: pass where ``raster <= max_value``.

    Nodata-aware: the raster is read masked, and masked / NaN / out-of-
    extent samples **never pass** (this fixes the China notebook's
    "NaN slope is not excluded" leak, and the geodata ``Mask`` behaviour
    where a -9999 ocean pixel passes ``< max_bound`` as suitable).

    .. warning:: Check the raster's declared nodata before trusting the
       strict policy.  A product that declares ``nodata=0`` (e.g. the
       China elevation layer) makes every genuine 0-value sample —
       sea-level land for an elevation raster, flat ground for a slope
       raster — read as masked and therefore FAIL.  The China regression
       notebook (04) quantifies this: it is a data-declaration problem;
       rewrite the raster's nodata tag (or pre-fill) rather than
       loosening the sampler.

    The raster is loaded into memory once at factory time — intended for
    the ~1 km terrain layers (slope, elevation), not for 30 m land cover.

    Parameters
    ----------
    raster_path : str
        Terrain raster (any CRS; sample points are transformed into it).
        Keep units explicit at the call site: slope thresholds are in the
        raster's own units (degrees for SRTM/GLO-30 products, percent for
        the Indonesia USGS product — see garuda.md section 4.3).
    max_value : float
        Samples with value strictly greater than this fail.
    min_value : float, optional
        If given, samples below this also fail.
    band : int, optional
        Raster band (default 1).

    Returns
    -------
    callable
        ``(lons, lats) -> bool`` array; True = passes.
    """
    with rasterio.open(raster_path) as src:
        tr = src.transform
        if tr.b != 0 or tr.d != 0:
            raise ValueError("rotated rasters are not supported")
        data = src.read(band, masked=True)
        crs = src.crs
    values = np.ma.filled(data.astype(np.float64), np.nan)
    h, w = values.shape
    inv = None
    if crs is not None and crs.to_epsg() != 4326:
        inv = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    a, c_off = tr.a, tr.c
    e, f_off = tr.e, tr.f

    def sampler(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if inv is not None:
            x, y = inv.transform(lons, lats)
        else:
            x, y = lons, lats
        cols = np.floor((x - c_off) / a).astype(np.int64)
        rows = np.floor((y - f_off) / e).astype(np.int64)
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.zeros(lons.shape, dtype=bool)
        if ok.any():
            v = values[rows[ok], cols[ok]]
            good = np.isfinite(v) & (v <= max_value)
            if min_value is not None:
                good &= v >= min_value
            out[ok] = good
        return out

    return sampler


def polygon_sampler(
    gdf,
    *,
    invert: bool = False,
    res_deg: float = 0.0025,
    all_touched: bool = True,
) -> ExclusionSampler:
    """Exclusion-sampler factory from polygons (e.g. protected areas).

    Polygons are rasterized once onto a ``res_deg`` lon/lat grid over
    their own bounds (the validated China-notebook approach: 0.0025 deg
    is ~250 m). By default polygons are **exclusion zones**: a sample
    passes when it is *not* inside any polygon; points outside the
    rasterized extent are trivially outside every polygon and pass.
    With ``invert=True`` the polygons are keep-in zones instead (a sample
    passes only *inside*; outside the extent fails).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or GeoSeries
        Polygons. Reprojected to EPSG:4326 if a CRS is set; assumed 4326
        (with a warning) otherwise.
    invert : bool, optional
        False (default): pass outside polygons. True: pass inside.
    res_deg : float, optional
        Rasterization resolution in degrees (default 0.0025 ~ 250 m).
    all_touched : bool, optional
        Passed to ``rasterio.features.rasterize`` (default True, as in the
        validated pipeline — conservative for exclusion zones).

    Returns
    -------
    callable
        ``(lons, lats) -> bool`` array; True = passes.
    """
    geoms = gdf.geometry if hasattr(gdf, "geometry") else gdf
    crs = getattr(geoms, "crs", None)
    if crs is None:
        logger.warning("polygon_sampler: input has no CRS; assuming EPSG:4326")
    elif crs.to_epsg() != 4326:
        geoms = geoms.to_crs("EPSG:4326")
    shapes = [g for g in geoms if g is not None and not g.is_empty]

    if not shapes:
        logger.warning("polygon_sampler: no non-empty geometries; sampler is a no-op")

        def trivial(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
            shape = np.asarray(lons).shape
            return np.zeros(shape, bool) if invert else np.ones(shape, bool)

        return trivial

    minx, miny, maxx, maxy = np.array([g.bounds for g in shapes]).T.reshape(4, -1)
    lon0, lat0 = float(minx.min()) - res_deg, float(miny.min()) - res_deg
    lon1, lat1 = float(maxx.max()) + res_deg, float(maxy.max()) + res_deg
    w = max(1, int(np.ceil((lon1 - lon0) / res_deg)))
    h = max(1, int(np.ceil((lat1 - lat0) / res_deg)))
    if w * h > 2_000_000_000:
        raise ValueError(
            f"polygon_sampler grid {w}x{h} at res_deg={res_deg} is too large; "
            "increase res_deg"
        )
    tr = rasterio.transform.from_bounds(lon0, lat0, lon1, lat1, w, h)
    inside = rasterio.features.rasterize(
        ((g, 1) for g in shapes),
        out_shape=(h, w),
        transform=tr,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype(bool)
    logger.info(
        "polygon_sampler: rasterized %d polygons to %dx%d at %g deg (%.1f%% inside)",
        len(shapes), w, h, res_deg, 100.0 * inside.mean(),
    )

    def sampler(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        cols = np.floor((lons - lon0) / res_deg).astype(np.int64)
        rows = np.floor((lat1 - lats) / res_deg).astype(np.int64)
        inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        hit = np.zeros(lons.shape, dtype=bool)
        if inb.any():
            hit[inb] = inside[rows[inb], cols[inb]]
        return hit if invert else ~hit

    return sampler
