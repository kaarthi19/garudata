"""Turbine power curves and solar-panel parameters for garudata.

Data provenance
---------------
All numbers below are extracted verbatim from the **geodata** package resource
library at the ``master`` branch (GeodataTools/geodata, GPL-3.0):

* ``src/geodata/resources/windturbine/Siemens_SWT_107_3600kW.yaml``
* ``src/geodata/resources/windturbine/Vestas_V112_3MW.yaml``
* ``src/geodata/resources/solarpanel/KANEKA.yaml``  (Bofinger-type model)
* ``src/geodata/resources/solarpanel/CSi.yaml``     (Huld-type model)

geodata is licensed GPL-3.0; this extraction retains that attribution.

Conventions
-----------
``TURBINES[name]``
    ``V``           power-curve wind speeds [m/s], non-decreasing; duplicated
                    breakpoints encode the cut-in/cut-out steps (e.g. the final
                    ``25, 25`` pair drops power to 0 exactly at cut-out).
    ``POW``         **per-unit capacity factor** at each ``V`` (original
                    yaml power in MW divided by the rated power ``P``);
                    values in [0, 1].  Multiply by ``P`` to recover MW.
    ``hub_height``  manufacturer hub height [m].
    ``P``           rated power [MW] (max of the yaml ``POW`` list).

``PANELS[name]``
    The yaml parameter set kept as-is (one dict per panel).  Note that
    despite the task shorthand "Huld coefficients", the two panels use two
    different geodata PV models: **KANEKA is a Bofinger-model panel**
    (A/B/C efficiency fit + D temperature coefficient) and **CSi is a
    Huld-model panel** (k_1..k_6 fitting parameters).  The ``model`` key
    disambiguates.  ``garudata.convert.pv_cf_poa`` only consumes the
    temperature coefficient (``D`` when present), so both entries are safe
    defaults there.

Notes
-----
Power-curve lookups must use ``np.interp(v, V, POW, left=0.0, right=0.0)``.
``scipy.interpolate.interp1d(..., fill_value="extrapolate")`` is forbidden:
with the duplicated final breakpoint it extrapolates a -inf slope above
cut-out (confirmed geodata dataset-refactor bug, ``model/wind/_base.py:137``).
With ``np.interp`` the duplicated abscissa evaluates to the *second* value at
the exact breakpoint, so v == 25.0 m/s yields CF = 0.0 (correct shutdown).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Wind turbines                                                               #
# --------------------------------------------------------------------------- #

# Siemens SWT 3.6-107 (onshore, 3.6 MW, 90 m hub).
# Source yaml: resources/windturbine/Siemens_SWT_107_3600kW.yaml @ master.
_SIEMENS_SWT_107_V: list[float] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 25.0, 25.0,
]
_SIEMENS_SWT_107_POW_MW: list[float] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.161, 0.351, 0.635, 1.026, 1.544,
    2.204, 2.910, 3.399, 3.567, 3.596, 3.6, 3.6, 0.0,
]
_SIEMENS_SWT_107_P_MW: float = 3.6

# Vestas V112 3.0 MW (offshore datasheet, 80 m hub).
# Source yaml: resources/windturbine/Vestas_V112_3MW.yaml @ master.
_VESTAS_V112_V: list[float] = [
    0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    25.0, 25.0,
]
_VESTAS_V112_POW_MW: list[float] = [
    0.000, 0.000, 0.005, 0.150, 0.300, 0.525, 0.905, 1.375, 1.950, 2.580,
    2.960, 3.050, 3.060, 3.060, 0.000,
]
_VESTAS_V112_P_MW: float = 3.06

TURBINES: dict[str, dict] = {
    "Siemens_SWT_107_3600kW": {
        "name": "Siemens SWT 107 3600 kW",
        "manufacturer": "Siemens",
        "V": _SIEMENS_SWT_107_V,
        # per-unit CF curve = POW_MW / P
        "POW": [p / _SIEMENS_SWT_107_P_MW for p in _SIEMENS_SWT_107_POW_MW],
        "hub_height": 90.0,
        "P": _SIEMENS_SWT_107_P_MW,
    },
    "Vestas_V112_3MW": {
        "name": "V112 3MW",
        "manufacturer": "Vestas",
        "V": _VESTAS_V112_V,
        "POW": [p / _VESTAS_V112_P_MW for p in _VESTAS_V112_POW_MW],
        "hub_height": 80.0,
        "P": _VESTAS_V112_P_MW,
    },
}

# --------------------------------------------------------------------------- #
# Solar panels                                                                #
# --------------------------------------------------------------------------- #

PANELS: dict[str, dict] = {
    # Bofinger-model thin-film panel (geodata resources/solarpanel/KANEKA.yaml).
    # efficiency = A + B*I + C*log(I); D is the temperature power coefficient.
    # Temperatures in K, radiation in W/m2, exactly as in the yaml.
    "KANEKA": {
        "model": "bofinger",
        "name": "Kaneka Hybrid",
        "manufacturer": "Kaneka",
        "threshold": 1,            # W/m2 below which output is zero
        "area": 1.22,              # m2
        "rated_production": 89.3,  # Wp
        "A": 0.0659164166836276,
        "B": -4.44310393547043e-06,
        "C": 0.0122044905275824,
        "D": -0.0035,              # temperature power coefficient [1/K]
        "NOCT": 318,               # K (= 45 degC)
        "Tstd": 298,               # K (STC module temperature)
        "Tamb": 293,               # K (NTC ambient temperature)
        "Intc": 800,               # W/m2 (NTC radiation)
        "ta": 0.9,                 # transmittance x absorptance
        "inverter_efficiency": 0.9,
    },
    # Huld-model crystalline-silicon panel (geodata resources/solarpanel/CSi.yaml,
    # after Huld 2010).  k_1..k_6 are the Huld fitting parameters.
    "CSi": {
        "model": "huld",
        "name": "CSi",
        "source": "Huld 2010",
        "efficiency": 0.1,
        "c_temp_amb": 1,           # panel temp coeff. of ambient temperature
        "c_temp_irrad": 0.035,     # K / (W/m2)
        "r_tamb": 293,             # K (20 degC reference ambient)
        "r_tmod": 298,             # K (25 degC reference module)
        "r_irradiance": 1000,      # W/m2 reference irradiance
        "k_1": -0.017162,
        "k_2": -0.040289,
        "k_3": -0.004681,
        "k_4": 0.000148,
        "k_5": 0.000169,
        "k_6": 0.000005,
        "inverter_efficiency": 0.9,
    },
}
