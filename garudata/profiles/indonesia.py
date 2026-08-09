"""Indonesia profile constants for garudata (interface spec §4.3, verbatim).

Every constant cites its provenance (GARUDA tool file:line, captive notebook
cell id, or GARUDA docs).  Values here are the single source of truth for
Indonesia runs — pipeline code must take them as parameters, never hardcode.

.. warning:: :data:`KLHK_SUITABLE_CLASSES` **is a deliberate ``None``
   placeholder** — the real KLHK land-cover class codes were never resolved
   (captive notebook cell ``039bce1d`` marks its ``[20, 21, 2010, 2014, 2500]``
   list as a TODO with guessed values mixing two code conventions).  Any
   function receiving ``None`` for the class list MUST raise (use
   :func:`require_klhk_classes`), never fall back silently: results computed
   with placeholder classes are meaningless.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Bounding boxes (W, S, E, N) in EPSG:4326                                    #
# --------------------------------------------------------------------------- #

#: Indonesia captive-RE domain: lon 95..141, lat -11..6
#: (captive notebook cell ``039bce1d``: XS=slice(95,141), YS=slice(-11,6)).
BBOX_INDONESIA: tuple[float, float, float, float] = (95.0, -11.0, 141.0, 6.0)

#: NTT / Timor default box (``tools/solar_resource_era5.py:101-102``).
BBOX_NTT: tuple[float, float, float, float] = (121.0, -11.0, 125.0, -8.5)

#: Timor GIS analysis extent (``docs/resource_siting.md:26``).
BBOX_TIMOR_GIS: tuple[float, float, float, float] = (123.2, -10.5, 125.3, -8.9)

# --------------------------------------------------------------------------- #
# Time zones (hours ahead of UTC)                                             #
# --------------------------------------------------------------------------- #

#: Indonesian time zones (``tools/solar_resource_era5.py:110-112``):
#: WIB (west) +7, WITA (central, incl. NTT — the tool default) +8, WIT (east) +9.
TZ_OFFSETS_HOURS: dict[str, int] = {"WIB": 7, "WITA": 8, "WIT": 9}

#: Default zone for the NTT/Timor village pipeline (WITA, +8).
TZ_DEFAULT: str = "WITA"

# --------------------------------------------------------------------------- #
# Land-cover suitability (captive workflow)                                   #
# --------------------------------------------------------------------------- #

#: The operational land layer is the GARUDA-convention SUITABILITY-SCORE
#: raster (``landcover_suitability.tif``: full-Indonesia box, EPSG:4326,
#: 10000x10000 float32, nodata=0), derived from the Indonesia land-cover
#: product by ``garuda/tools/candidate_land.py`` scoring — NOT raw KLHK class
#: codes.  Score semantics (candidate_land.py ``--min-suitable``):
SUITABILITY_SCORES: dict[int, str] = {
    0: "unsuitable / nodata (conflated in this product)",
    2: "suitable incl. shrub & savanna",
    3: "suitable, settlement/farm/bare land only (strict)",
}

#: Default keep-set for the captive mask: score >= 2 (includes shrub/savanna),
#: matching candidate_land.py's default. Use the strict variant for
#: sensitivity runs.
SUITABLE_SCORES: list[int] = [2, 3]
SUITABLE_SCORES_STRICT: list[int] = [3]

#: KLHK land-cover class codes to KEEP as developable.
#:
#: ``None`` on purpose.  Only needed for a RAW KLHK class raster — the
#: operational path uses :data:`SUITABLE_SCORES` on the suitability-score
#: raster above instead.  The captive notebook (cell ``039bce1d``) ships the
#: placeholder ``[20, 21, 2010, 2014, 2500]`` explicitly marked TODO — the
#: mixed 2-digit/4-digit codes suggest they were guessed from two different
#: KLHK code conventions.  The siting readme only states the principle:
#: exclude forest, residential and water.  If a raw KLHK product is ever used
#: directly, supply its real codes and call :func:`require_klhk_classes`.
KLHK_SUITABLE_CLASSES: Optional[list[int]] = None


def require_klhk_classes(
    classes: Optional[Sequence[int]] = None,
) -> list[int]:
    """Return a validated KLHK keep-class list, refusing the None placeholder.

    Parameters
    ----------
    classes : sequence of int, optional
        Explicit class codes; defaults to :data:`KLHK_SUITABLE_CLASSES`.

    Returns
    -------
    list of int
        The resolved class codes.

    Raises
    ------
    ValueError
        If the resolved value is ``None`` (the unresolved placeholder) or
        empty.  This is intentional: silently defaulting would reproduce the
        notebook's meaningless placeholder mask.
    """
    resolved = classes if classes is not None else KLHK_SUITABLE_CLASSES
    if resolved is None:
        raise ValueError(
            "KLHK_SUITABLE_CLASSES is unresolved (None). The real KLHK "
            "land-cover class codes must be supplied before the captive land "
            "mask can be built — the notebook's [20, 21, 2010, 2014, 2500] "
            "is an explicit placeholder TODO (cell 039bce1d) and must NOT be "
            "used. Principle per the siting readme: exclude forest, "
            "residential and water classes."
        )
    resolved = [int(c) for c in resolved]
    if not resolved:
        raise ValueError("KLHK suitable-class list is empty")
    return resolved


# --------------------------------------------------------------------------- #
# Slope thresholds (units explicit in the names!)                             #
# --------------------------------------------------------------------------- #

#: Captive solar: exclude cells with slope > 5 PERCENT
#: (USGS/EarthExplorer percent-slope raster; notebook cell ``039bce1d`` +
#: docx Table 1).  NOTE: percent, not degrees — 5% is about 2.86 deg.
SLOPE_MAX_SOLAR_PCT: float = 5.0

#: Captive wind: exclude cells with slope > 20 PERCENT (same sources).
SLOPE_MAX_WIND_PCT: float = 20.0

#: NTT/village candidate pipeline uses a DIFFERENT convention: slope <= 15
#: DEGREES on the Copernicus GLO-30-derived raster (``candidate_land.py:43``).
#: Kept here so the percent-vs-degrees split stays explicit (spec §4.3).
SLOPE_MAX_NTT_CANDIDATE_DEG: float = 15.0

# --------------------------------------------------------------------------- #
# Packing densities [MW/km2]                                                  #
# --------------------------------------------------------------------------- #

#: Three defensible PV densities and one wind density are in play — expose as
#: parameters, never hardcode (spec §4.3):
#: - ``captive_solar`` 30.0 and ``captive_wind`` 3.9: captive notebook cell
#:   ``039bce1d`` ("from the siting README");
#: - ``ntt_candidate`` 62.0: ``tools/candidate_land.py:44`` default;
#: - ``usable_land`` 50.0: ``tools/ntt/usable_land.py:57-58`` (~2 ha/MWp),
#:   the current provenance-correct Timor cap source.
PACKING: dict[str, float] = {
    "captive_solar": 30.0,
    "captive_wind": 3.9,
    "ntt_candidate": 62.0,
    "usable_land": 50.0,
}

# --------------------------------------------------------------------------- #
# Facility buffers (captive workflow)                                         #
# --------------------------------------------------------------------------- #

#: Buffer radius around each industrial facility [km]
#: (notebook cell ``30efdc64``: ``BUFFER_KM = 10``).
FACILITY_BUFFER_KM: float = 10.0

#: Equal-area metric CRS used to build the buffers before converting back to
#: EPSG:4326 (notebook cell ``30efdc64``).
FACILITY_BUFFER_CRS: str = "EPSG:6933"

# --------------------------------------------------------------------------- #
# Connection-cost constants (tools/ntt/costs.py:24-47)                        #
# --------------------------------------------------------------------------- #

#: Fixed grid-connection cost [Rp] (Rp 150 million).
CONNECTION_FIXED_RP: float = 150e6

#: MV feeder cost per km [Rp/km] (Rp 400 million per km).
CONNECTION_RP_PER_KM: float = 400e6

#: Exchange rate [Rp per USD].
FX_RP_PER_USD: float = 16_000.0

#: Real discount rate for connection-cost annualization (10%).
CONNECTION_DISCOUNT_RATE: float = 0.10

#: Annualization horizon [years].
CONNECTION_LIFETIME_YEARS: int = 30


def crf(rate: float, years: int) -> float:
    """Capital recovery factor: ``r(1+r)^n / ((1+r)^n - 1)``.

    Parameters
    ----------
    rate : float
        Real discount rate per year (e.g. 0.10).
    years : int
        Annualization horizon in years.

    Returns
    -------
    float
        The capital recovery factor (per-year fraction of present cost).
    """
    if rate == 0:
        return 1.0 / years
    growth = (1.0 + rate) ** years
    return rate * growth / (growth - 1.0)


#: CRF at 10% over 30 years (~0.10608), as used by ``tools/ntt/costs.py``:
#: ``Cost_per_yr = (150e6 + hubdist_km*400e6) / 16000 * CRF(10%, 30y)``.
CONNECTION_CRF: float = crf(CONNECTION_DISCOUNT_RATE, CONNECTION_LIFETIME_YEARS)

#: ``Max_Connect_MW = max(peak_mw * 1.5, 0.02)`` (``tools/ntt/costs.py:43-47``).
MAX_CONNECT_PEAK_FACTOR: float = 1.5
MAX_CONNECT_FLOOR_MW: float = 0.02

#: Fallback hub distance for coordinate-less villages [km] (``costs.py:83-104``).
FALLBACK_HUBDIST_KM: float = 10.0

# --------------------------------------------------------------------------- #
# Technology defaults (captive workflow, notebook cell ``039bce1d``)          #
# --------------------------------------------------------------------------- #

#: Solar panel in :data:`garudata.resources.PANELS` (fixed tilt, KANEKA).
PANEL: str = "KANEKA"

#: Wind turbine in :data:`garudata.resources.TURBINES`
#: (Siemens SWT-107, ~3.6 MW, 90 m hub).
TURBINE: str = "Siemens_SWT_107_3600kW"

#: Representative weather year for the captive analysis.
YEAR_CAPTIVE: int = 2018

# --------------------------------------------------------------------------- #
# Regression anchors — expected magnitudes for validation runs               #
# --------------------------------------------------------------------------- #

#: Known-good reference numbers to regression-test any reimplementation
#: against (spec §4.3).  Ranges are (low, high) tuples.
REGRESSION: dict = {
    # NTT mean solar CF and full-load hours (docs/solar_resource_geodata.md:68-69)
    "ntt_mean_solar_cf": (0.18, 0.22),
    "ntt_full_load_hours": (1600.0, 1900.0),
    # Timor usable-land caps (data_indonesia/DATA_PROVENANCE.md:45): total and
    # median of village Max_Cap_MW after the shared-allocation fix (the old
    # buffer mode summed to 2,472 GW = 13x the actual land).
    "timor_caps_total_GW": 233.0,
    "timor_caps_median_MW": 234.0,
    # Belu 81-village run optimum, million USD (docs/resource_siting.md:54-58);
    # caps non-binding by ~1,460x.
    "belu_81_village_optimum_MUSD": 8.3597634,
    # Coordinate-less villages fall back to the NTT centroid; ERA5 ~28 km
    # cells collapse them to a few dozen distinct profiles
    # (data_indonesia/DATA_PROVENANCE.md:58-66).
    "unsited_villages": (153, 780),
    "unsited_centroid_latlon": (-9.69766, 124.36616),
}
