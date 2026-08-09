"""Per-polygon attribution of cell-level quantities.

Port of the validated China-project "figure 4" method
(``geodata_figures_kag.ipynb`` section 4, per-PA blocked-potential
league table):

1. build a lon/lat box for every non-zero grid cell;
2. overlay-intersect the cell boxes with the polygons in an equal-area
   metric CRS (default EPSG:6933);
3. allocate each cell's value to the intersecting polygons proportionally
   to intersection area — where polygons overlap each other, both count
   toward the denominator, i.e. overlaps are split, never double counted;
4. verify conservation: the value allocated across all polygons must equal
   the summed value of every cell that touches at least one polygon
   (reproduces cell totals exactly, to a tolerance).

This is the identity-preserving alternative to geodata's
``Mask.add_shape_layer``, which dissolves all polygons into one layer.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

__all__ = ["attribute_to_polygons"]


def _as_latlon_2d(da: xr.DataArray) -> xr.DataArray:
    """Normalise a 2-D DataArray to ascending (lat, lon) dims."""
    ren = {}
    if "y" in da.dims:
        ren["y"] = "lat"
    if "x" in da.dims:
        ren["x"] = "lon"
    if ren:
        da = da.rename(ren)
    if set(da.dims) != {"lat", "lon"}:
        raise ValueError(
            f"cell_values must be 2-D with dims (lat, lon) or (y, x); got {da.dims}"
        )
    return da.transpose("lat", "lon").sortby("lat").sortby("lon")


def attribute_to_polygons(
    cell_values: xr.DataArray,
    polygons: gpd.GeoDataFrame,
    *,
    cell_size_deg: float = 0.25,
    metric_crs: str = "EPSG:6933",
    value_name: str | None = None,
    conserve_rtol: float = 1e-6,
) -> pd.DataFrame:
    """Attribute a cell-level quantity to individual polygons.

    Parameters
    ----------
    cell_values : xarray.DataArray
        2-D grid of the quantity to attribute (any additive quantity:
        TWh, km2, MW...), dims ``(lat, lon)`` or ``(y, x)`` with lon/lat
        coordinates at **cell centers**. NaN and zero cells are skipped.
    polygons : geopandas.GeoDataFrame
        Target polygons. Invalid geometries are repaired with
        ``make_valid``; a missing CRS is assumed EPSG:4326 (warned).
    cell_size_deg : float, optional
        Grid cell edge in degrees (default 0.25, the ERA5 grid); cell
        boxes are centered on the coordinates with half this size on each
        side.
    metric_crs : str, optional
        Equal-area CRS for intersection areas (default EPSG:6933).
    value_name : str, optional
        Name for the output value column; defaults to
        ``cell_values.name`` or ``"value"``.
    conserve_rtol : float, optional
        Relative tolerance of the conservation check (default 1e-6).

    Returns
    -------
    pandas.DataFrame
        Indexed like ``polygons`` with columns:

        - ``<value_name>`` — allocated quantity (0.0 for polygons that
          intersect no non-zero cell);
        - ``intersection_km2`` — total cell-box intersection area;
        - ``n_cells`` — number of distinct cells contributing.

        ``DataFrame.attrs`` carries ``total_cell_value`` (all non-zero
        cells), ``allocated_value`` and ``unallocated_value`` (cells with
        no polygon overlap), plus ``metric_crs`` and ``cell_size_deg``.

    Raises
    ------
    RuntimeError
        If the conservation check fails: allocated total differs from the
        summed value of polygon-covered cells beyond ``conserve_rtol``.
    """
    da = _as_latlon_2d(cell_values)
    vals = np.asarray(da.values, dtype=float)
    lats = np.asarray(da["lat"].values, dtype=float)
    lons = np.asarray(da["lon"].values, dtype=float)
    vname = value_name or (str(cell_values.name) if cell_values.name else "value")
    if vname == "intersection_km2" or vname == "n_cells":
        raise ValueError(f"value_name {vname!r} clashes with an output column")

    iy, ix = np.nonzero(np.isfinite(vals) & (vals != 0.0))
    total_cell_value = float(vals[iy, ix].sum()) if iy.size else 0.0
    logger.info(
        "attribute_to_polygons: %d non-zero cells (total %g), %d polygons",
        iy.size, total_cell_value, len(polygons),
    )

    out = pd.DataFrame(
        {vname: 0.0, "intersection_km2": 0.0, "n_cells": 0},
        index=polygons.index,
    )
    out.attrs.update(
        total_cell_value=total_cell_value,
        allocated_value=0.0,
        unallocated_value=total_cell_value,
        metric_crs=metric_crs,
        cell_size_deg=cell_size_deg,
    )
    if iy.size == 0 or len(polygons) == 0:
        return out

    half = cell_size_deg / 2.0
    cells = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(iy.size),
            "_cell_value": vals[iy, ix],
        },
        geometry=[
            box(lons[j] - half, lats[i] - half, lons[j] + half, lats[i] + half)
            for i, j in zip(iy, ix)
        ],
        crs="EPSG:4326",
    )

    polys = polygons[[polygons.geometry.name]].copy()
    polys["_poly_pos"] = np.arange(len(polys))
    if polys.crs is None:
        logger.warning("polygons have no CRS; assuming EPSG:4326")
        polys = polys.set_crs("EPSG:4326")
    invalid = ~polys.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.info("repairing %d invalid polygon geometries with make_valid", n_invalid)
        polys.loc[invalid, polys.geometry.name] = polys.loc[
            invalid, polys.geometry.name
        ].apply(make_valid)

    inter = gpd.overlay(
        cells.to_crs(metric_crs),
        polys.to_crs(metric_crs),
        how="intersection",
        keep_geom_type=True,
    )
    inter["ia_km2"] = inter.geometry.area / 1e6
    inter = inter[inter["ia_km2"] > 0]
    if inter.empty:
        logger.info("no cell/polygon intersections; nothing allocated")
        return out

    # Each polygon's share of the total polygon-intersection area within
    # the cell (overlapping polygons split the cell's value, never
    # double counting it).
    denom = inter.groupby("cell_id")["ia_km2"].transform("sum")
    inter["_alloc"] = inter["_cell_value"] * inter["ia_km2"] / denom

    per_poly = inter.groupby("_poly_pos").agg(
        _alloc=("_alloc", "sum"),
        intersection_km2=("ia_km2", "sum"),
        n_cells=("cell_id", "nunique"),
    )
    pos = per_poly.index.to_numpy()
    out.iloc[pos, out.columns.get_loc(vname)] = per_poly["_alloc"].to_numpy()
    out.iloc[pos, out.columns.get_loc("intersection_km2")] = per_poly[
        "intersection_km2"
    ].to_numpy()
    out.iloc[pos, out.columns.get_loc("n_cells")] = per_poly["n_cells"].to_numpy()

    allocated = float(per_poly["_alloc"].sum())
    covered = float(
        cells.loc[cells["cell_id"].isin(inter["cell_id"].unique()), "_cell_value"].sum()
    )
    if not np.isclose(allocated, covered, rtol=conserve_rtol, atol=1e-9):
        raise RuntimeError(
            "conservation check failed: allocated "
            f"{allocated!r} != covered-cell total {covered!r} "
            f"(rtol={conserve_rtol})"
        )
    out.attrs["allocated_value"] = allocated
    out.attrs["unallocated_value"] = total_cell_value - allocated
    logger.info(
        "allocated %g of %g (%.2f%%); unallocated (cells with no polygon overlap) %g",
        allocated,
        total_cell_value,
        100.0 * allocated / total_cell_value if total_cell_value else 0.0,
        total_cell_value - allocated,
    )
    return out
