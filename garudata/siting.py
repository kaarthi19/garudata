"""Facility buffers, per-facility RE potential, polygon->point allocation.

Implements the siting half of the GARUDA mini-package contract
(garuda.md section 4.1 items 8-9):

- :func:`facility_buffers` — points CSV -> {name: 10-km buffer polygon},
  buffered in an equal-area metric CRS (captive notebook cell 30efdc64);
- :func:`facility_potential` — the captive notebook's per-facility
  aggregation (indonesia.md section c8: cos-lat cell areas,
  nearest-reindex of coarse CF onto the fine suitability grid, masked
  mean over suitable cells), with its fragilities fixed (section d):
  suitability is passed explicitly instead of read from mutable
  ``Mask.merged_mask`` state, cell spacing comes from the full grid so
  degenerate 1-row/column clips cannot break it, and a NaN mean CF
  yields zero capacity with an explicit flag;
- :func:`allocate_to_points` — the three polygon-to-point allocation
  modes with the semantics of ``tools/ntt/usable_land.py`` (shared) and
  ``tools/resource_siting.py`` (buffer, nearest);
- :func:`hub_distance` — nearest-substation distance and name
  (``resource_siting.py`` lines 109-114 semantics).

No dependency on the geodata package.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "facility_buffers",
    "facility_potential",
    "allocate_to_points",
    "hub_distance",
    "LAT_ALIASES",
    "LON_ALIASES",
    "NAME_ALIASES",
]

#: km per degree of latitude; same convention as the captive notebook.
KM_PER_DEG = 111.32

# Column aliases, matching resource_siting.py (lat/lon) and the id aliases
# of solar_resource_geodata.py / the captive facility CSV (garuda.md 2.1).
LAT_ALIASES = ("lat", "Latitude", "latitude", "LAT")
LON_ALIASES = ("lon", "Longitude", "longitude", "LON", "lng")
NAME_ALIASES = ("Facility", "name", "Name", "Village", "village", "id", "ID", "Site", "site")

_CHUNK = 2048  # polygon-chunk size for the distance-matrix modes


def _pick(columns, aliases, given, what):
    if given is not None:
        if given not in columns:
            raise ValueError(f"{what} column {given!r} not in {list(columns)}")
        return given
    col = next((c for c in aliases if c in columns), None)
    if col is None:
        raise ValueError(f"no {what} column found (tried {aliases})")
    return col


def _as_latlon_2d(da: xr.DataArray, what: str) -> xr.DataArray:
    """Normalise a DataArray to ascending 2-D (lat, lon)."""
    ren = {}
    if "y" in da.dims:
        ren["y"] = "lat"
    if "x" in da.dims:
        ren["x"] = "lon"
    if ren:
        # Cutout-style arrays carry lon/lat *alias* coordinates on the
        # x/y dims (cutout.normalize_coords); drop them so the dim rename
        # cannot collide with an existing non-dim coordinate.
        clash = [c for c in ren.values() if c in da.coords and c not in da.dims]
        if clash:
            da = da.drop_vars(clash)
        da = da.rename(ren)
    if set(da.dims) != {"lat", "lon"}:
        raise ValueError(
            f"{what} must be 2-D with dims (lat, lon) or (y, x); got {da.dims} "
            "(reduce e.g. with .mean('time') first)"
        )
    return da.transpose("lat", "lon").sortby("lat").sortby("lon")


def facility_buffers(
    points_df: pd.DataFrame,
    radius_km: float = 10.0,
    *,
    name_col: str | None = None,
    lat_col: str | None = None,
    lon_col: str | None = None,
    metric_crs: str = "EPSG:6933",
) -> dict[str, BaseGeometry]:
    """Build per-facility circular buffers in EPSG:4326.

    Reproduces captive notebook cell ``30efdc64``: points are projected to
    an equal-area metric CRS (default EPSG:6933), buffered by
    ``radius_km``, and returned as lon/lat polygons keyed by facility name.

    .. note:: EPSG:6933 preserves **area**, not shape: at the equator a
       10 km "circle" is an ~11.5 x 8.7 km ground ellipse (area still
       exactly ``pi * radius_km**2``).  This matches the notebook
       convention; pass a local UTM zone as ``metric_crs`` if truly
       circular buffers matter more than byte-reproduction.

    Parameters
    ----------
    points_df : pandas.DataFrame
        One row per facility with name + latitude/longitude columns.
        Columns are auto-detected from the standard aliases
        (:data:`NAME_ALIASES`, :data:`LAT_ALIASES`, :data:`LON_ALIASES`)
        unless given explicitly.
    radius_km : float, optional
        Buffer radius in km (default 10, the captive-RE convention).
    name_col, lat_col, lon_col : str, optional
        Explicit column names, overriding alias detection.
    metric_crs : str, optional
        CRS in which the metric buffering is done (default EPSG:6933).

    Returns
    -------
    dict of str to shapely geometry
        ``{facility_name: buffer_polygon}`` in EPSG:4326 — the shape dict
        consumed by :func:`facility_potential` (and by geodata's
        ``Mask.extract_shapes``, whose only safe input form is exactly
        such a plain dict).
    """
    name = _pick(points_df.columns, NAME_ALIASES, name_col, "name")
    lat = _pick(points_df.columns, LAT_ALIASES, lat_col, "latitude")
    lon = _pick(points_df.columns, LON_ALIASES, lon_col, "longitude")

    df = points_df[[name, lat, lon]].copy()
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    ok = df[lat].notna() & df[lon].notna()
    if (~ok).any():
        logger.warning("facility_buffers: dropping %d row(s) without coordinates", int((~ok).sum()))
    df = df[ok]
    dup = df[name].duplicated()
    if dup.any():
        logger.warning(
            "facility_buffers: %d duplicate name(s) dropped (keeping first): %s",
            int(dup.sum()), sorted(df.loc[dup, name].astype(str).unique())[:5],
        )
        df = df[~dup]

    g = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326"
    )
    buffered = g.to_crs(metric_crs).buffer(radius_km * 1000.0).to_crs("EPSG:4326")
    out = dict(zip(df[name].astype(str), buffered))
    logger.info("facility_buffers: %d buffer(s) of %g km built in %s", len(out), radius_km, metric_crs)
    return out


def facility_potential(
    suitability: xr.DataArray,
    cf: xr.DataArray,
    buffers: Mapping[str, BaseGeometry],
    packing_mw_per_km2: float,
    *,
    km_per_deg: float = KM_PER_DEG,
) -> pd.DataFrame:
    """Suitable area, mean CF and deployable capacity per facility buffer.

    For each buffer polygon (EPSG:4326, from :func:`facility_buffers`):

    - clip the fine suitability grid to the polygon (cell centers inside);
    - cell areas from lat/lon spacing with cos-lat correction
      (``dlon*111.32*cos(lat) x dlat*111.32`` km2 — captive notebook cell
      ``1d4a2147``), the spacing taken from the **full** grid so a
      degenerate one-row/one-column clip cannot produce a bogus spacing;
    - suitable area = sum of cell areas weighted by the suitability value
      (0/1 masks and fractional suitability both supported; NaN = 0);
    - mean CF = the coarse CF grid nearest-reindexed onto the clip
      (nearest-neighbour upsampling of e.g. 0.25-degree ERA5 CF), averaged
      unweighted over cells with suitability > 0 (the notebook's masked
      mean; implicitly area-weights each coarse CF cell by its suitable
      fine-pixel count);
    - capacity_MW = suitable_area_km2 x ``packing_mw_per_km2``.

    Fixes over the notebook (indonesia.md section d): suitability is an
    explicit argument (no reliance on ``Mask.merged_mask`` mutation
    ordering, so solar/wind passes cannot silently swap masks), and a
    facility with no suitable cells — or whose CF window is all-NaN —
    gets ``capacity_MW = 0`` and ``flagged = True`` instead of a silent
    NaN sorted to the end.

    Parameters
    ----------
    suitability : xarray.DataArray
        Fine-grid suitability, 0/1 or fraction in [0, 1], dims
        ``(lat, lon)`` or ``(y, x)``.
    cf : xarray.DataArray
        Capacity-factor surface on a (typically coarser) lon/lat grid,
        e.g. an annual-mean ERA5 CF. Must be 2-D.
    buffers : mapping of str to shapely geometry
        ``{facility: polygon in EPSG:4326}``.
    packing_mw_per_km2 : float
        Packing density (captive conventions: solar 30, wind 3.9;
        expose per profile — garuda.md section 4.3).
    km_per_deg : float, optional
        km per degree (default 111.32, the shared convention).

    Returns
    -------
    pandas.DataFrame
        Columns ``Facility, suitable_area_km2, mean_CF, capacity_MW,
        flagged``; sorted by ``mean_CF`` descending, flagged rows last.
    """
    suit = _as_latlon_2d(suitability, "suitability")
    cfa = _as_latlon_2d(cf, "cf")

    lat_v = suit["lat"].values.astype(float)
    lon_v = suit["lon"].values.astype(float)
    if lat_v.size < 2 or lon_v.size < 2:
        raise ValueError("suitability grid must have at least 2 points per axis")
    dlat = float(np.median(np.diff(lat_v)))
    dlon = float(np.median(np.diff(lon_v)))

    rows = []
    for fac, poly in buffers.items():
        minx, miny, maxx, maxy = poly.bounds
        i0, i1 = np.searchsorted(lon_v, [minx, maxx])
        j0, j1 = np.searchsorted(lat_v, [miny, maxy])
        i1 = min(i1 + 1, lon_v.size)
        j1 = min(j1 + 1, lat_v.size)
        sub = suit.isel(lat=slice(j0, j1), lon=slice(i0, i1))

        suit_area = 0.0
        mean_cf = float("nan")
        if sub.sizes["lat"] > 0 and sub.sizes["lon"] > 0:
            lon_g, lat_g = np.meshgrid(sub["lon"].values, sub["lat"].values)
            inside = shapely.contains_xy(poly, lon_g, lat_g)
            sv = np.nan_to_num(np.asarray(sub.values, dtype=float), nan=0.0) * inside
            aw = dlon * km_per_deg * np.cos(np.deg2rad(sub["lat"].values.astype(float)))
            area2d = np.outer(aw * (dlat * km_per_deg), np.ones(sub.sizes["lon"]))
            suit_area = float((area2d * sv).sum())
            m = sv > 0
            if m.any():
                cf_near = cfa.reindex_like(sub, method="nearest")
                vals = np.asarray(cf_near.values, dtype=float)[m]
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    mean_cf = float(finite.mean())

        flagged = not np.isfinite(mean_cf)
        capacity = 0.0 if flagged else suit_area * packing_mw_per_km2
        if flagged:
            logger.warning(
                "facility_potential: %r has no suitable cells with valid CF "
                "(suitable_area_km2=%.3f) -> capacity_MW=0, flagged", fac, suit_area
            )
        rows.append(
            {
                "Facility": fac,
                "suitable_area_km2": suit_area,
                "mean_CF": mean_cf,
                "capacity_MW": capacity,
                "flagged": flagged,
            }
        )

    out = (
        pd.DataFrame(rows, columns=["Facility", "suitable_area_km2", "mean_CF", "capacity_MW", "flagged"])
        .sort_values("mean_CF", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return out


def allocate_to_points(
    polygons_gdf: gpd.GeoDataFrame,
    points_df: pd.DataFrame,
    *,
    value_col: str,
    mode: str = "shared",
    radius_km: float | None = 10.0,
    metric_crs: str = "EPSG:6933",
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> pd.DataFrame:
    """Allocate a polygon-level quantity (e.g. MW, area) to points.

    The three modes and their documented failure envelopes
    (garuda.md section 2.7 / 4.1 item 8; usable_land.py lines 146-162):

    ``shared`` (default; ``usable_land.py`` semantics — the canonical
        algorithm)
        Each polygon's value is split **equally among all points within**
        ``radius_km`` of the polygon centroid. Total-preserving: no
        hectare is double counted, and no point with land nearby is left
        at zero. Polygons with no point in range go to the unallocated
        remainder.
    ``buffer`` (``resource_siting.py`` buffer mode)
        Per point, sum the value of polygons intersecting a
        ``radius_km`` buffer, each weighted by its **in-buffer area
        fraction** (area-clipped — summing whole polygons over-counts
        badly when polygons are large). CAVEAT — deliberate double
        counting: the same land is counted once for *every* point in
        range, so per-point figures do NOT sum to the land that exists
        (on Timor this inflated totals to 2,472 GW against 233 GW of
        actual land, DATA_PROVENANCE.md:45). Physically it answers "what
        could *this* site develop within R km", not "how much is there".
    ``nearest`` (centroid Voronoi; ``resource_siting.py`` allocate mode)
        Each polygon goes entirely to the nearest point (by centroid),
        within ``radius_km`` if given (``radius_km=None`` = unlimited
        Voronoi). Partition, no double counting — but starves any point
        that is never the closest (178 of 627 Timor villages got nothing
        while a neighbour a few hundred metres away got hundreds of MW).

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        Polygons carrying ``value_col``. A missing CRS is assumed
        EPSG:4326 (warned).
    points_df : pandas.DataFrame
        Points with lat/lon columns (aliases auto-detected). Rows with
        missing coordinates get NaN results — never silently dropped, so
        the output aligns row-for-row with the input.
    value_col : str
        Numeric column of ``polygons_gdf`` to allocate.
    mode : {"shared", "buffer", "nearest"}, optional
    radius_km : float or None, optional
        Radius in km (default 10). Required for "shared" and "buffer";
        None is allowed only for "nearest" (pure Voronoi).
    metric_crs : str, optional
        Projected CRS for distances/areas (default EPSG:6933; pass
        EPSG:3857 to reproduce the legacy QGIS runs).
    lat_col, lon_col : str, optional
        Explicit coordinate columns.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``points_df`` with columns ``<value_col>`` and
        ``area_km2`` (the allocated polygon land area under the same
        weights). ``DataFrame.attrs`` carries ``mode``, ``radius_km``,
        ``unallocated_value`` and ``unallocated_area_km2`` (polygons
        reached by no point; for "buffer" this is untouched-polygon
        value, and allocated totals may exceed the existing total by
        design).
    """
    if mode not in ("shared", "buffer", "nearest"):
        raise ValueError(f"unknown mode {mode!r} (use 'shared', 'buffer' or 'nearest')")
    if radius_km is None and mode != "nearest":
        raise ValueError(f"radius_km is required for mode={mode!r}")
    if value_col not in polygons_gdf.columns:
        raise ValueError(f"polygons_gdf has no column {value_col!r}")
    if value_col == "area_km2":
        raise ValueError("value_col 'area_km2' clashes with the output area column")

    lat = _pick(points_df.columns, LAT_ALIASES, lat_col, "latitude")
    lon = _pick(points_df.columns, LON_ALIASES, lon_col, "longitude")

    polys = polygons_gdf
    if polys.crs is None:
        logger.warning("allocate_to_points: polygons have no CRS; assuming EPSG:4326")
        polys = polys.set_crs("EPSG:4326")
    polys_m = polys.to_crs(metric_crs)
    pval = pd.to_numeric(polys_m[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    parea = polys_m.geometry.area.to_numpy(dtype=float) / 1e6  # km2
    total_val, total_area = float(pval.sum()), float(parea.sum())

    out = pd.DataFrame({value_col: np.nan, "area_km2": np.nan}, index=points_df.index)
    out.attrs.update(
        mode=mode,
        radius_km=radius_km,
        metric_crs=metric_crs,
        unallocated_value=total_val,
        unallocated_area_km2=total_area,
    )
    lat_n = pd.to_numeric(points_df[lat], errors="coerce")
    lon_n = pd.to_numeric(points_df[lon], errors="coerce")
    valid = (lat_n.notna() & lon_n.notna()).to_numpy()
    if not valid.any() or len(polys_m) == 0:
        logger.warning("allocate_to_points: no located points or no polygons; all-NaN/empty result")
        return out

    pts = (
        gpd.GeoSeries(gpd.points_from_xy(lon_n[valid], lat_n[valid]), crs="EPSG:4326")
        .to_crs(metric_crs)
    )
    npts = len(pts)
    val_tot = np.zeros(npts)
    area_tot = np.zeros(npts)

    if mode == "buffer":
        R = float(radius_km) * 1000.0
        sidx = polys_m.sindex
        geoms = polys_m.geometry
        touched = np.zeros(len(polys_m), dtype=bool)
        for k, geom in enumerate(pts.values):
            buf = geom.buffer(R)
            cand = np.asarray(sidx.query(buf, predicate="intersects"))
            if cand.size == 0:
                continue
            inter = geoms.iloc[cand].intersection(buf).area.to_numpy(dtype=float)
            pos = inter > 0
            cand, inter = cand[pos], inter[pos]
            if cand.size == 0:
                continue
            frac = inter / (parea[cand] * 1e6)
            val_tot[k] = float((pval[cand] * frac).sum())
            area_tot[k] = float(inter.sum()) / 1e6
            touched[cand] = True
        unalloc_val = float(pval[~touched].sum())
        unalloc_area = float(parea[~touched].sum())
    else:
        cent = polys_m.geometry.centroid
        px = cent.x.to_numpy(dtype=float)
        py = cent.y.to_numpy(dtype=float)
        vx = pts.x.to_numpy(dtype=float)
        vy = pts.y.to_numpy(dtype=float)
        r2 = np.inf if radius_km is None else (float(radius_km) * 1000.0) ** 2
        unalloc_val = 0.0
        unalloc_area = 0.0
        # Chunked over polygons: the full distance matrix would be
        # n_polygons x n_points (~14 M entries on the real NTT layer).
        for lo in range(0, len(px), _CHUNK):
            hi = min(lo + _CHUNK, len(px))
            d2 = (px[lo:hi, None] - vx[None, :]) ** 2 + (py[lo:hi, None] - vy[None, :]) ** 2
            v = pval[lo:hi]
            a = parea[lo:hi]
            if mode == "nearest":
                j = d2.argmin(axis=1)
                within = d2[np.arange(hi - lo), j] <= r2
                np.add.at(val_tot, j[within], v[within])
                np.add.at(area_tot, j[within], a[within])
                unalloc_val += float(v[~within].sum())
                unalloc_area += float(a[~within].sum())
            else:  # shared
                near = d2 <= r2
                counts = near.sum(axis=1)
                has = counts > 0
                unalloc_val += float(v[~has].sum())
                unalloc_area += float(a[~has].sum())
                if has.any():
                    val_tot += (near * np.where(has, v / np.maximum(counts, 1), 0.0)[:, None]).sum(axis=0)
                    area_tot += (near * np.where(has, a / np.maximum(counts, 1), 0.0)[:, None]).sum(axis=0)

    out.loc[out.index[valid], value_col] = val_tot
    out.loc[out.index[valid], "area_km2"] = area_tot
    out.attrs["unallocated_value"] = unalloc_val
    out.attrs["unallocated_area_km2"] = unalloc_area
    logger.info(
        "allocate_to_points[%s, r=%s km]: allocated %s=%.1f (of %.1f existing) over "
        "%d located point(s); unallocated %.1f%s",
        mode, radius_km, value_col, float(val_tot.sum()), total_val, npts, unalloc_val,
        "; NOTE buffer mode double-counts shared land by design" if mode == "buffer" else "",
    )
    return out


def hub_distance(
    points_df: pd.DataFrame,
    substations_gdf: gpd.GeoDataFrame,
    *,
    metric_crs: str = "EPSG:6933",
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> pd.DataFrame:
    """Distance to (and name of) the nearest substation per point.

    Semantics of ``resource_siting.py`` lines 109-114: nearest substation
    by spatial index in a projected CRS; the substation name column is
    auto-detected from ``HubName | name | Name | NAME`` with a
    ``sub_<i>`` fallback. Maps to GARUDA as: hubdist_km -> connection
    cost, nearest substation -> Zone.

    .. warning:: The default ``metric_crs`` EPSG:6933 is equal-**area**
       (standard parallel 30 deg), not equidistant: near the equator its
       planar distances are ~13% short east-west and ~15% long
       north-south.  For distance fidelity in Indonesia prefer
       ``metric_crs="EPSG:3857"`` (near-true scale at the equator, and
       what the legacy QGIS analysis used) or the local UTM zone.

    Parameters
    ----------
    points_df : pandas.DataFrame
        Points with lat/lon columns (aliases auto-detected). Rows with
        missing coordinates get NaN/None results, never dropped.
    substations_gdf : geopandas.GeoDataFrame
        Substation point layer; a missing CRS is assumed EPSG:4326.
    metric_crs : str, optional
        Projected CRS for distances (default EPSG:6933; use EPSG:3857 to
        reproduce the legacy QGIS analysis).
    lat_col, lon_col : str, optional
        Explicit coordinate columns.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``points_df`` with columns ``hubdist_km`` (float)
        and ``hub_name`` (str or None).
    """
    if len(substations_gdf) == 0:
        raise ValueError("substations_gdf is empty")
    lat = _pick(points_df.columns, LAT_ALIASES, lat_col, "latitude")
    lon = _pick(points_df.columns, LON_ALIASES, lon_col, "longitude")

    subs = substations_gdf
    if subs.crs is None:
        logger.warning("hub_distance: substations have no CRS; assuming EPSG:4326")
        subs = subs.set_crs("EPSG:4326")
    subs_m = subs.to_crs(metric_crs).reset_index(drop=True)
    name_col = next((c for c in ("HubName", "name", "Name", "NAME") if c in subs_m.columns), None)

    out = pd.DataFrame({"hubdist_km": np.nan, "hub_name": None}, index=points_df.index)
    lat_n = pd.to_numeric(points_df[lat], errors="coerce")
    lon_n = pd.to_numeric(points_df[lon], errors="coerce")
    valid = (lat_n.notna() & lon_n.notna()).to_numpy()
    if not valid.any():
        logger.warning("hub_distance: no points with coordinates")
        return out

    pts = (
        gpd.GeoSeries(gpd.points_from_xy(lon_n[valid], lat_n[valid]), crs="EPSG:4326")
        .to_crs(metric_crs)
    )
    idx, dist = subs_m.sindex.nearest(pts, return_all=False, return_distance=True)
    hub_pos = np.empty(len(pts), dtype=np.int64)
    d_m = np.empty(len(pts), dtype=float)
    hub_pos[idx[0]] = idx[1]
    d_m[idx[0]] = dist

    names = (
        subs_m[name_col].astype(str).to_numpy()[hub_pos]
        if name_col
        else np.array([f"sub_{j}" for j in hub_pos])
    )
    out.loc[out.index[valid], "hubdist_km"] = d_m / 1000.0
    out.loc[out.index[valid], "hub_name"] = names
    return out
