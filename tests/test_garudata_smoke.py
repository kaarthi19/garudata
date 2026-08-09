"""Functional smoke tests for the garudata package (synthetic data only).

Runs two ways:
  pytest tests/                                   # after `pip install -e .[dev]`
  python tests/test_garudata_smoke.py             # standalone, no pytest needed

All tests are offline and self-contained; temporary rasters/CSVs are written
next to this file and are gitignored.
"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("garudata") is None:
    sys.path.insert(0, str(REPO_ROOT))

HAVE_PVLIB = importlib.util.find_spec("pvlib") is not None

import numpy as np
import pandas as pd
import xarray as xr

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()[-1]
        RESULTS.append((name, "FAIL", tb))
        print(f"FAIL  {name}: {tb}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# 0. import cleanliness
# --------------------------------------------------------------------------- #

def test_import_silent():
    import subprocess
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "import garudata"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout == "", f"import printed to stdout: {r.stdout!r}"
    assert r.stderr == "", f"import printed to stderr: {r.stderr!r}"


def test_poa_import_error():
    # Only provable in an env WITHOUT pvlib: pv_cf_poa must raise a clear
    # ImportError naming the missing dependency. With pvlib installed the
    # optionality claim is exercised by the real-data validation instead.
    if HAVE_PVLIB:
        print("SKIP (pvlib installed — optional-dependency error path not testable)")
        return
    from garudata import convert
    ds = xr.Dataset(
        {"influx_direct": (("time", "y", "x"), np.ones((2, 1, 1))),
         "influx_diffuse": (("time", "y", "x"), np.ones((2, 1, 1)))},
        coords={"time": pd.date_range("2018-01-01", periods=2, freq="h"),
                "y": [0.0], "x": [0.0]},
    )
    try:
        convert.pv_cf_poa(ds)
        raise AssertionError("pv_cf_poa should raise ImportError without pvlib")
    except ImportError as e:
        assert "pvlib" in str(e)


# --------------------------------------------------------------------------- #
# a. grids
# --------------------------------------------------------------------------- #

def test_grids():
    from garudata import grids
    lons, lats = grids.era5_grid((95.0, -11.0, 141.0, 6.0))
    assert lons.size == int(round((141 - 95) / 0.25)) + 1 == 185, lons.size
    assert lats.size == int(round((6 - -11) / 0.25)) + 1 == 69, lats.size
    assert lons[0] == 95.0 and lons[-1] == 141.0
    assert lats[0] == 6.0 and lats[-1] == -11.0  # descending, ERA5 convention
    assert np.allclose(np.mod(lons, 0.25), 0.0)
    assert np.allclose(np.mod(np.abs(lats), 0.25), 0.0)
    assert np.all(np.diff(lons) > 0) and np.all(np.diff(lats) < 0)

    areas = grids.cell_areas_km2(lons, lats)
    assert areas.shape == (69, 185)
    assert np.all(areas > 0)
    # cos-lat: max at lat=0, strictly decreasing away from equator
    col = areas[:, 0]
    i0 = int(np.argmax(col))
    assert lats[i0] == 0.0, f"max area at lat={lats[i0]}"
    assert np.all(np.diff(col[i0:]) < 0)          # southward from equator
    assert np.all(np.diff(col[:i0 + 1]) > 0)      # approaching equator from north
    # sanity magnitude: 0.25 deg cell at equator ~ (27.8 km)^2 ~ 773 km2
    assert abs(col[i0] - 773.0) < 10.0, col[i0]

    lat_idx, lon_idx = grids.nearest_cell(lons, lats, [95.1, 140.9], [5.9, -10.9])
    assert lons[lon_idx[0]] == 95.0 and lons[lon_idx[1]] == 141.0
    assert lats[lat_idx[0]] == 6.0 and lats[lat_idx[1]] == -11.0


# --------------------------------------------------------------------------- #
# b. cutout.canonicalize
# --------------------------------------------------------------------------- #

def _raw_ds():
    t = pd.date_range("2018-01-01", periods=4, freq="h")
    y = [0.25, 0.0]
    x = [100.0, 100.25]
    shape = (4, 2, 2)

    def v(val, units):
        return xr.DataArray(np.full(shape, float(val)), dims=("time", "y", "x"),
                            attrs={"units": units})

    ds = xr.Dataset(
        {
            "ssrd": v(3600.0, "J m**-2"),
            "fdir": v(1800.0, "J m**-2"),
            "tisr": v(4500.0, "J m**-2"),
            "ssr": v(3240.0, "J m**-2"),   # albedo = (3600-3240)/3600 = 0.1
            "t2m": v(298.15, "K"),
            "u100": v(3.0, "m s**-1"),
            "v100": v(4.0, "m s**-1"),
            "fsr": v(0.1, "m"),
            "d2m": v(290.0, "K"),
        },
        coords={"time": t, "y": y, "x": x},
    )
    return ds


def test_canonicalize():
    from garudata import cutout
    raw = _raw_ds()
    ds = cutout.canonicalize(raw)
    for name in ("influx_direct", "influx_diffuse", "influx_toa", "albedo",
                 "temperature", "wnd100m", "roughness", "dewpoint_temperature"):
        assert name in ds.data_vars, f"missing {name}: {list(ds.data_vars)}"
    # J/m2 hourly accumulation -> W/m2 division convention:
    # influx_direct = fdir/3600, influx_diffuse = (ssrd-fdir)/3600 -> total = ssrd/3600 = 1 W/m2
    assert np.allclose(ds["influx_direct"].values, 0.5)
    assert np.allclose(ds["influx_diffuse"].values, 0.5)
    total = ds["influx_direct"] + ds["influx_diffuse"]
    assert np.allclose(total.values, 1.0), "ssrd=3600 J must give 1 W/m2 total influx"
    assert np.allclose(ds["influx_toa"].values, 4500.0 / 3600.0)
    assert ds["influx_direct"].attrs["units"] == "W m**-2"
    assert np.allclose(ds["albedo"].values, 0.1)
    assert np.allclose(ds["temperature"].values, 298.15)
    assert np.allclose(ds["wnd100m"].values, 5.0)  # sqrt(3^2+4^2)
    # raw vars gone
    for name in ("ssrd", "ssr", "fdir", "tisr", "u100", "v100", "t2m", "fsr", "d2m"):
        assert name not in ds.data_vars
    assert cutout.is_canonical(ds)
    # idempotent: second call must not divide by 3600 again
    ds2 = cutout.canonicalize(ds)
    assert np.allclose(ds2["influx_direct"].values, 0.5)
    assert np.allclose(ds2["influx_diffuse"].values, 0.5)
    xr.testing.assert_identical(ds, ds2)
    # original raw ds untouched
    assert "ssrd" in raw.data_vars and np.allclose(raw["ssrd"].values, 3600.0)


def test_cutout_coords():
    from garudata import cutout
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 4)))},
        coords={"valid_time": pd.date_range("2018-01-01", periods=2, freq="h"),
                "latitude": [1.0, 0.5, 0.0], "longitude": [10.0, 10.25, 10.5, 10.75]},
    )
    nds = cutout.normalize_coords(ds)
    assert "time" in nds.dims and "x" in nds.dims and "y" in nds.dims
    assert "lon" in nds.coords and "lat" in nds.coords
    sub = cutout.sel_bounds(nds, (10.25, 0.0, 10.5, 0.5))  # descending lat handled
    assert sub.sizes["x"] == 2 and sub.sizes["y"] == 2


# --------------------------------------------------------------------------- #
# c. convert: pv_cf_ghi and wind_cf
# --------------------------------------------------------------------------- #

def test_pv_cf_ghi():
    from garudata import convert
    n = 72
    t = pd.date_range("2018-01-01", periods=n, freq="h")
    ghi = np.zeros((n, 1, 1))
    # diurnal peak at UTC hour 4 (== local noon in WITA UTC+8)
    for d in range(3):
        ghi[d * 24 + 4, 0, 0] = 1000.0
    ds = xr.Dataset(
        {
            "ssrd": (("time", "y", "x"), ghi * 3600.0),  # J/m2 accumulations
            "t2m": (("time", "y", "x"), np.full((n, 1, 1), 25.0 + 273.15)),
        },
        coords={"time": t, "y": [0.0], "x": [100.0]},
    )
    ds["ssrd"].attrs["units"] = "J m**-2"
    ds["t2m"].attrs["units"] = "K"

    # gamma=0 isolates the GHI term: CF == PR exactly at GHI=1000, any T
    cf0 = convert.pv_cf_ghi(ds, gamma=0.0)
    assert float(cf0.isel(y=0, x=0, time=4)) == 0.80, float(cf0.isel(y=0, x=0, time=4))

    # default gamma/noct: reference model value at GHI=1000, T2m=25C:
    # Tcell = 25 + 25/800*1000 = 56.25; CF = 0.8*(1 - 0.004*31.25) = 0.7
    cf = convert.pv_cf_ghi(ds)
    v = float(cf.isel(y=0, x=0, time=4))
    assert abs(v - 0.7) < 1e-12, v
    assert float(cf.isel(y=0, x=0, time=0)) == 0.0
    assert cf.dims == ("time", "y", "x")
    assert float(cf.min()) >= 0.0 and float(cf.max()) <= 1.0

    # tz roll: +8 h shifts argmax from UTC hour 4 to local hour 12
    cf_loc = convert.pv_cf_ghi(ds, gamma=0.0, tz_offset_hours=8)
    day0 = cf_loc.isel(y=0, x=0).values[:24]
    assert int(np.argmax(day0)) == 12, int(np.argmax(day0))


def test_wind_cf():
    from garudata import convert
    speeds = np.array([0.0, 5.0, 10.0, 14.0, 24.999, 25.0, 30.0])
    n = speeds.size
    t = pd.date_range("2018-01-01", periods=n, freq="h")
    ds = xr.Dataset(
        {"wnd100m": (("time", "y", "x"), speeds.reshape(n, 1, 1))},
        coords={"time": t, "y": [0.0], "x": [100.0]},
    )
    cf = convert.wind_cf(ds, hub_height=100.0)  # no extrapolation
    got = cf.isel(y=0, x=0).values
    assert got[0] == 0.0                                   # calm
    assert abs(got[1] - 0.351 / 3.6) < 1e-12               # 5 m/s point
    assert abs(got[2] - 2.910 / 3.6) < 1e-12               # 10 m/s point
    assert got[3] == 1.0                                   # rated
    assert got[4] > 0.99                                   # just below cut-out
    assert got[5] == 0.0, got[5]                           # exactly at cut-out (dup breakpoint)
    assert got[6] == 0.0, got[6]                           # above cut-out: EXACTLY 0, no -inf
    assert np.all(np.isfinite(got))

    # hub-height extrapolation monotone in hub height (power law, no roughness)
    base = xr.Dataset(
        {"wnd100m": (("time", "y", "x"), np.full((1, 1, 1), 8.0))},
        coords={"time": t[:1], "y": [0.0], "x": [100.0]},
    )
    cfs_pow = [convert.wind_cf(base, hub_height=h).values.item() for h in (50, 80, 100, 120, 150)]
    assert all(a <= b for a, b in zip(cfs_pow, cfs_pow[1:])), cfs_pow
    # log law with roughness present
    base_z = base.copy()
    base_z["fsr"] = (("time", "y", "x"), np.full((1, 1, 1), 0.1))
    cfs_log = [convert.wind_cf(base_z, hub_height=h).values.item() for h in (50, 80, 100, 120, 150)]
    assert all(a <= b for a, b in zip(cfs_log, cfs_log[1:])), cfs_log
    assert cfs_pow[2] == cfs_log[2]  # at 100 m both are the untouched 100 m speed


# --------------------------------------------------------------------------- #
# d. land.class_area_by_cell
# --------------------------------------------------------------------------- #

def test_land():
    import rasterio
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    from garudata import land

    scratch = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(scratch, "_synth_landcover.tif")

    # raster footprint = lon 0..0.5, lat 0..0.5 in EPSG:6933 metric coords
    tf = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x0, y0 = tf.transform(0.0, 0.0)
    x1, y1 = tf.transform(0.5, 0.5)
    npx = 100
    transform = from_bounds(x0, y0, x1, y1, npx, npx)
    data = np.empty((npx, npx), dtype=np.uint8)
    data[:, :50] = 10   # class "west"
    data[:, 50:] = 20   # class "east"
    with rasterio.open(
        path, "w", driver="GTiff", height=npx, width=npx, count=1,
        dtype="uint8", crs="EPSG:6933", transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)

    total_km2 = abs((x1 - x0) * (y1 - y0)) / 1e6
    grid_lons = [0.125, 0.375]
    grid_lats = [0.125, 0.375]
    class_map = {10: "west", 20: "east"}

    # window must be a multiple of block or each window's sub-block remainder
    # rows/cols are dropped (documented validated-pipeline behavior; with the
    # real defaults 8192/10 the loss is 0.02%, with 64/10 it would be 19%).
    ds = land.class_area_by_cell(path, class_map, grid_lons, grid_lats,
                                 window=50, block=10)
    got_total = float(ds["area"].sum())
    assert abs(got_total - total_km2) / total_km2 < 0.01, (got_total, total_km2)
    per_class = ds["area"].sum(dim=("lat", "lon")).values
    assert np.allclose(per_class, total_km2 / 2.0, rtol=0.01), per_class
    # west class must live in the western cells only
    west = ds["area"].sel(land_class="west")
    assert float(west.sel(lon=0.375).sum()) < 1e-9 * total_km2
    assert float(ds["cell_area"].min()) > 0
    frac = ds["fraction"].sum("land_class")
    assert np.all((frac.values > 0.85) & (frac.values < 1.15)), frac.values

    # exclusion sampler rejecting the southern half halves every class area
    keep_north = lambda lons, lats: np.asarray(lats) > 0.25
    ds2 = land.class_area_by_cell(path, class_map, grid_lons, grid_lats,
                                  exclusion_samplers=[keep_north],
                                  window=50, block=10)
    got2 = float(ds2["area"].sum())
    assert abs(got2 - total_km2 / 2.0) / total_km2 < 0.01, (got2, total_km2 / 2.0)
    assert "area_unmasked" in ds2
    assert abs(float(ds2["area_unmasked"].sum()) - total_km2) / total_km2 < 0.01
    per_class2 = ds2["area"].sum(dim=("lat", "lon")).values
    assert np.allclose(per_class2, total_km2 / 4.0, rtol=0.02), per_class2

    # suitable_area_by_cell collapse
    ds3 = land.suitable_area_by_cell(path, [10, 20], grid_lons, grid_lats,
                                     window=50, block=10)
    assert "land_class" not in ds3.dims
    assert abs(float(ds3["area"].sum()) - total_km2) / total_km2 < 0.01


def test_terrain_sampler():
    import rasterio
    from rasterio.transform import from_bounds
    from garudata import land

    scratch = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(scratch, "_synth_slope.tif")
    slope = np.full((10, 10), 3.0, dtype=np.float32)
    slope[:, 5:] = 30.0
    slope[0, 0] = -9999.0  # nodata pixel
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 10, 10), nodata=-9999.0,
    ) as dst:
        dst.write(slope, 1)
    s = land.terrain_sampler(path, max_value=15.0)
    lons = np.array([0.25, 0.75, 0.05, 5.0])
    lats = np.array([0.5, 0.5, 0.95, 0.5])
    got = s(lons, lats)
    assert got.tolist() == [True, False, False, False], got  # nodata & out-of-extent fail


# --------------------------------------------------------------------------- #
# e. attribute
# --------------------------------------------------------------------------- #

def test_attribute():
    import geopandas as gpd
    from shapely.geometry import box
    from garudata import attribute

    vals = np.array([[1.0, 2.0], [3.0, 4.0]])  # rows = lat asc
    da = xr.DataArray(vals, coords={"lat": [0.125, 0.375], "lon": [0.125, 0.375]},
                      dims=("lat", "lon"), name="twh")
    polys = gpd.GeoDataFrame(
        {"name": ["bottom", "top"]},
        geometry=[box(0.0, 0.0, 0.5, 0.25), box(0.0, 0.25, 0.5, 0.5)],
        crs="EPSG:4326",
    )
    out = attribute.attribute_to_polygons(da, polys)
    assert abs(out.loc[0, "twh"] - 3.0) < 1e-9, out
    assert abs(out.loc[1, "twh"] - 7.0) < 1e-9, out
    assert abs(out.attrs["allocated_value"] - 10.0) < 1e-9
    assert abs(out.attrs["unallocated_value"]) < 1e-9
    assert out["n_cells"].tolist() == [2, 2]

    # split coverage: two polygons each covering half of ONE cell -> value split,
    # cell total conserved exactly; other cells unallocated
    polys2 = gpd.GeoDataFrame(
        {"name": ["left_half", "right_half"]},
        geometry=[box(0.0, 0.0, 0.125, 0.25), box(0.125, 0.0, 0.25, 0.25)],
        crs="EPSG:4326",
    )
    out2 = attribute.attribute_to_polygons(da, polys2)
    assert abs(out2.loc[0, "twh"] - 0.5) < 1e-6, out2
    assert abs(out2.loc[1, "twh"] - 0.5) < 1e-6, out2
    assert abs(out2.attrs["allocated_value"] - 1.0) < 1e-9
    assert abs(out2.attrs["unallocated_value"] - 9.0) < 1e-9


# --------------------------------------------------------------------------- #
# f. siting
# --------------------------------------------------------------------------- #

def test_facility_buffers_potential():
    from garudata import siting

    pts = pd.DataFrame({"Facility": ["F1"], "lat": [0.0], "lon": [0.0]})
    bufs = siting.facility_buffers(pts, radius_km=10.0)
    assert set(bufs) == {"F1"}
    poly = bufs["F1"]
    minx, miny, maxx, maxy = poly.bounds
    # EPSG:6933 is equal-AREA (std parallel 30 deg): at the equator the 10 km
    # buffer is an ~11.5 x 8.7 km ground ellipse, area exactly pi*100 km2.
    assert 0.19 < (maxx - minx) < 0.22, poly.bounds
    assert 0.14 < (maxy - miny) < 0.17, poly.bounds
    area_km2 = poly.area * 111.32 ** 2  # deg2 -> km2 at the equator
    assert abs(area_km2 - np.pi * 100.0) / (np.pi * 100.0) < 0.02, area_km2

    # fine suitability grid: all ones, 0.01 deg spacing
    ax = np.arange(-0.2, 0.2001, 0.01)
    suit = xr.DataArray(np.ones((ax.size, ax.size)),
                        coords={"lat": ax, "lon": ax}, dims=("lat", "lon"))
    cf = xr.DataArray(np.full((3, 3), 0.2),
                      coords={"lat": [-0.25, 0.0, 0.25], "lon": [-0.25, 0.0, 0.25]},
                      dims=("lat", "lon"))
    out = siting.facility_potential(suit, cf, bufs, packing_mw_per_km2=30.0)
    row = out.iloc[0]
    expect_area = np.pi * 10.0 ** 2
    assert abs(row["suitable_area_km2"] - expect_area) / expect_area < 0.05, row
    assert abs(row["mean_CF"] - 0.2) < 1e-12
    assert abs(row["capacity_MW"] - row["suitable_area_km2"] * 30.0) < 1e-9
    assert not row["flagged"]

    # all-NaN CF window -> flagged, zero capacity (not silent NaN)
    out_nan = siting.facility_potential(suit, cf * np.nan, bufs, packing_mw_per_km2=30.0)
    r = out_nan.iloc[0]
    assert bool(r["flagged"]) and r["capacity_MW"] == 0.0
    assert np.isnan(r["mean_CF"])


def test_allocate_to_points():
    import geopandas as gpd
    from shapely.geometry import box
    from garudata import siting

    # polygon centered between two points ~5.5 km apart; a far polygon out of reach
    near_poly = box(0.02, -0.005, 0.03, 0.005)   # centroid (0.025, 0)
    far_poly = box(1.0, 1.0, 1.01, 1.01)
    polys = gpd.GeoDataFrame({"MW": [100.0, 7.0]}, geometry=[near_poly, far_poly],
                             crs="EPSG:4326")
    pts = pd.DataFrame({"name": ["P1", "P2", "P3"],
                        "lat": [0.0, 0.0, np.nan],
                        "lon": [0.0, 0.05, np.nan]})

    shared = siting.allocate_to_points(polys, pts, value_col="MW", mode="shared",
                                       radius_km=10.0)
    got = shared["MW"].to_numpy()
    assert abs(got[0] - 50.0) < 1e-9 and abs(got[1] - 50.0) < 1e-9, got
    assert np.isnan(got[2])  # missing coords -> NaN, never dropped
    # conservation: allocations sum to the value of polygons within reach
    assert abs(np.nansum(got) - 100.0) < 1e-9
    assert abs(shared.attrs["unallocated_value"] - 7.0) < 1e-9

    near = siting.allocate_to_points(polys, pts, value_col="MW", mode="nearest",
                                     radius_km=10.0)
    gn = near["MW"].to_numpy()
    # nearest starves: whole polygon to the closer point, the other gets 0
    assert (abs(gn[0] - 100.0) < 1e-9 and gn[1] == 0.0), gn
    assert abs(near.attrs["unallocated_value"] - 7.0) < 1e-9

    buf = siting.allocate_to_points(polys, pts, value_col="MW", mode="buffer",
                                    radius_km=10.0)
    gb = buf["MW"].to_numpy()
    # buffer double-counts: the same polygon fully inside both buffers
    assert abs(gb[0] - 100.0) < 0.5 and abs(gb[1] - 100.0) < 0.5, gb
    assert np.nansum(gb) > 150.0  # documented over-counting
    assert abs(buf.attrs["unallocated_value"] - 7.0) < 1e-9

    # docstring documents the failure envelopes
    doc = siting.allocate_to_points.__doc__
    assert "double count" in doc or "double-count" in doc.replace("counting", "count") or "double" in doc
    assert "starves" in doc


def test_hub_distance():
    import geopandas as gpd
    from garudata import siting

    subs = gpd.GeoDataFrame({"name": ["A", "B"]},
                            geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0]),
                            crs="EPSG:4326")
    pts = pd.DataFrame({"lat": [0.0, 0.0], "lon": [0.1, 0.9]})
    out = siting.hub_distance(pts, subs)
    assert out["hub_name"].tolist() == ["A", "B"], out
    # default EPSG:6933 is equal-area, NOT equidistant: E-W distances at the
    # equator come out ~13% short (0.1 deg -> ~9.65 km instead of 11.13 km)
    assert 9.0 < out["hubdist_km"].iloc[0] < 10.5, out
    # EPSG:3857 is near-true scale at the equator: 0.1 deg -> ~11.13 km
    out2 = siting.hub_distance(pts, subs, metric_crs="EPSG:3857")
    assert 10.8 < out2["hubdist_km"].iloc[0] < 11.5, out2
    assert out2["hub_name"].tolist() == ["A", "B"]


# --------------------------------------------------------------------------- #
# g. writers
# --------------------------------------------------------------------------- #

def test_writers_hourly():
    from garudata import writers

    scratch = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(scratch, "_writer_out")
    idx = pd.date_range("2023-01-01", periods=8760, freq="h", name="time")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"village_A": rng.uniform(0, 1, 8760), "B": rng.uniform(0, 1, 8760)},
        index=idx,
    )
    hourly_path, annual_path = writers.write_village_cf_hourly(df, out_dir)
    hourly = pd.read_csv(hourly_path)
    assert list(hourly.columns) == ["time", "village_A", "village_B"], hourly.columns
    assert len(hourly) == 8760
    annual = pd.read_csv(annual_path)
    assert list(annual.columns) == ["village", "mean_cf", "full_load_hours"]
    assert annual["village"].tolist() == ["A", "B"]
    for _, r in annual.iterrows():
        assert abs(r["full_load_hours"] - r["mean_cf"] * 8760.0) < 1e-6
    assert abs(annual.loc[0, "mean_cf"] - df["village_A"].mean()) < 1e-9


def test_writers_sample_points():
    from garudata import writers

    t = pd.date_range("2018-01-01", periods=24, freq="h")
    lats = np.array([0.0, 0.25])
    lons = np.array([100.0, 100.25])
    vals = np.zeros((24, 2, 2))
    vals[:, 1, 1] = 0.5  # cell (0.25, 100.25)
    cf = xr.DataArray(vals, coords={"time": t, "y": lats, "x": lons},
                      dims=("time", "y", "x"), name="cf")
    pts = pd.DataFrame({"village": ["v1", "v2"], "lat": [0.24, 0.01],
                        "lon": [100.26, 100.01]})
    out = writers.sample_points(cf, pts, utc_offset_hours=0)
    assert list(out.columns) == ["village_v1", "village_v2"]
    assert np.allclose(out["village_v1"], 0.5)
    assert np.allclose(out["village_v2"], 0.0)
    assert out.index.name == "time" and len(out) == 24


def test_writers_captive():
    from garudata import writers

    scratch = os.path.dirname(os.path.abspath(__file__))
    # exactly what facility_potential returns (incl. the flagged column)
    solar = pd.DataFrame({
        "Facility": ["F1", "F2"],
        "suitable_area_km2": [10.0, 5.0],
        "mean_CF": [0.21, 0.19],
        "capacity_MW": [300.0, 150.0],
        "flagged": [False, False],
    })
    wind = pd.DataFrame({
        "Facility": ["F1", "F2"],
        "suitable_area_km2": [8.0, 4.0],
        "mean_CF": [0.31, 0.29],
        "capacity_MW": [31.2, 15.6],
        "flagged": [False, False],
    })
    path = writers.write_captive_potential(solar, wind, 2018,
                                           os.path.join(scratch, "_writer_out"))
    got = pd.read_csv(path)
    want = ["Facility", "suitable_area_km2", "solar_CF", "solar_MW",
            "wind_CF", "wind_MW"]
    assert list(got.columns) == want, f"column order: {list(got.columns)} != {want}"
    assert got["Facility"].tolist() == ["F1", "F2"]
    assert abs(got.loc[0, "wind_MW"] - 31.2) < 1e-9
    assert path.endswith("captive_potential_2018.csv")


TESTS = [
    ("import: clean/silent", test_import_silent),
    ("convert.pv_cf_poa ImportError w/o pvlib", test_poa_import_error),
    ("grids: era5_grid + cell_areas + nearest_cell", test_grids),
    ("cutout.canonicalize: derived vars, W/m2, idempotent", test_canonicalize),
    ("cutout: normalize_coords + sel_bounds", test_cutout_coords),
    ("convert.pv_cf_ghi: CF==PR, ref value, tz roll", test_pv_cf_ghi),
    ("convert.wind_cf: cut-in/rated/cut-out, monotone hub", test_wind_cf),
    ("land.class_area_by_cell: totals + exclusion halving", test_land),
    ("land.terrain_sampler: nodata never passes", test_terrain_sampler),
    ("attribute: exact conservation + overlap split", test_attribute),
    ("siting.facility_buffers + facility_potential", test_facility_buffers_potential),
    ("siting.allocate_to_points: shared/nearest/buffer", test_allocate_to_points),
    ("siting.hub_distance", test_hub_distance),
    ("writers: hourly+annual schema, FLH", test_writers_hourly),
    ("writers.sample_points", test_writers_sample_points),
    ("writers.write_captive_potential column order", test_writers_captive),
]

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    for name, fn in TESTS:
        check(name, fn)
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
