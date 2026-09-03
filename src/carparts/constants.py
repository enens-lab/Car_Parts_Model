"""Canonical taxonomies + name normalisation. Everything user-facing speaks snake_case."""
from __future__ import annotations

import re

# ----------------------------------------------------------------------------- exterior (carparts-seg)
# Index order == the Ultralytics carparts-seg.yaml order (0..22). Kept verbatim for comparability.
EXTERIOR_CLASSES: list[str] = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light", "back_light",
    "back_right_door", "back_right_light", "front_bumper", "front_door", "front_glass", "front_left_door",
    "front_left_light", "front_light", "front_right_door", "front_right_light", "hood", "left_mirror",
    "object", "right_mirror", "tailgate", "trunk", "wheel",
]

# Side-agnostic product taxonomy: merges the L/R variants into their generic parent and drops the
# catch-all "object" label. Used by the `exterior_seg_clean` recipe.
EXTERIOR_CLEAN_MAP: dict[str, str | None] = {
    "back_left_door": "back_door", "back_right_door": "back_door",
    "front_left_door": "front_door", "front_right_door": "front_door",
    "back_left_light": "back_light", "back_right_light": "back_light",
    "front_left_light": "front_light", "front_right_light": "front_light",
    "left_mirror": "mirror", "right_mirror": "mirror",
    "object": None,
}

# ----------------------------------------------------------------------------- engine bay (Kaggle, 26)
# Index order == the Kaggle "Car Engine Bay Images with YOLO Annotations" class ids (0..25),
# with the two upstream typos fixed ("Housinig", "Housng").
ENGINE_BAY_CLASSES: list[str] = [
    "inverter_coolant_reservoir",     # 0
    "battery",                        # 1
    "radiator_cap",                   # 2
    "windshield_wiper_fluid",         # 3
    "fuse_box",                       # 4
    "power_steering_reservoir",       # 5
    "brake_fluid",                    # 6
    "engine_oil_fill_cap",            # 7
    "engine_oil_dipstick",            # 8
    "air_filter_cover",               # 9
    "abs_unit",                       # 10
    "alternator",                     # 11
    "engine_coolant_reservoir",       # 12
    "radiator",                       # 13
    "air_filter",                     # 14
    "engine_cover",                   # 15
    "cold_air_intake",                # 16
    "clutch_fluid_reservoir",         # 17
    "transmission_oil_dipstick",      # 18
    "intercooler_coolant_reservoir",  # 19
    "oil_filter_housing",             # 20
    "atf_oil_reservoir",              # 21
    "cabin_air_filter_housing",       # 22
    "secondary_coolant_reservoir",    # 23
    "electric_motor",                 # 24
    "oil_filter",                     # 25
]

_ENGINE_BAY_ALIASES: dict[str, str] = {
    # upstream typos / spelling variants seen on Kaggle + Roboflow re-uploads
    "oil_filter_housinig": "oil_filter_housing",
    "cabin_air_filter_housng": "cabin_air_filter_housing",
    "engine_oil_dip_stick": "engine_oil_dipstick",
    "transmission_oil_dip_stick": "transmission_oil_dipstick",
    "windshield_washer_fluid": "windshield_wiper_fluid",
    "windscreen_washer": "windshield_wiper_fluid",
    "brake_fluid_reservoir": "brake_fluid",
    "fuse-box": "fuse_box",
    "abs": "abs_unit",
}


# ----------------------------------------------------------------------------- parts catalog (isolated parts)
# Photos of a single component (catalog / used-parts style). Source: Roboflow "car parts" by Used auto parts
# classification (Public Domain, 50 classes, same taxonomy as Kaggle "50 Types of Car Parts", Apache 2.0) plus two
# CC BY 4.0 engine-internals sets that add connecting_rod and gear. Spelling fixed ("CARBERATOR").
PARTS_CATALOG_CLASSES: list[str] = [
    # engine core
    "engine_block", "cylinder_head", "piston", "connecting_rod", "crankshaft", "camshaft", "engine_valve",
    "valve_lifter", "oil_pan",
    # transmission & clutch
    "transmission", "torque_converter", "clutch_plate", "pressure_plate", "gear", "shift_knob",
    # engine accessories / fuel / ignition / cooling / exhaust
    "starter", "alternator", "water_pump", "fuel_injector", "carburetor", "spark_plug", "ignition_coil",
    "distributor", "thermostat", "oil_filter", "oil_pressure_sensor", "oxygen_sensor", "air_compressor",
    "radiator", "radiator_fan", "radiator_hose", "overflow_tank", "muffler",
    # electrical / body / interior
    "battery", "fuse_box", "headlight", "tail_light", "side_mirror", "spoiler", "gas_cap", "rim", "radio",
    "instrument_cluster", "window_regulator",
    # brakes & suspension / steering
    "brake_caliper", "brake_pad", "brake_rotor", "vacuum_brake_booster", "coil_spring", "leaf_spring",
    "lower_control_arm", "idler_arm",
]

# The user's powertrain definition (engine -> transmission/clutch -> driveline -> EV), restricted to what open
# data supports today. Driveline (driveshaft, u_joint, cv_joint, differential, axle, hub, transfer_case),
# flywheel, timing components, valve_body and EV traction parts have NO open datasets yet — see docs/datasets.md.
POWERTRAIN_CORE_CLASSES: list[str] = [
    "engine_block", "cylinder_head", "piston", "connecting_rod", "crankshaft", "camshaft", "engine_valve",
    "valve_lifter", "oil_pan", "transmission", "torque_converter", "clutch_plate", "pressure_plate", "gear",
]
POWERTRAIN_ACCESSORY_CLASSES: list[str] = [
    "starter", "alternator", "water_pump", "fuel_injector", "carburetor", "spark_plug", "ignition_coil",
    "distributor", "thermostat", "oil_filter", "oil_pressure_sensor", "oxygen_sensor",
]
POWERTRAIN_MISSING_CLASSES: list[str] = [  # wanted, no open data found (Sept 2026) — needs collection/labelling
    "flywheel_flexplate", "timing_chain_belt", "gearset", "valve_body", "transfer_case", "driveshaft",
    "universal_joint", "cv_joint", "differential", "axle_shaft", "wheel_hub", "traction_motor",
    "hv_battery_pack", "inverter",
]

_PARTS_CATALOG_ALIASES: dict[str, str] = {
    "carberator": "carburetor", "headlights": "headlight", "taillights": "tail_light", "tail_lights": "tail_light",
    "cam_shaft": "camshaft", "crank_shaft": "crankshaft", "connecting_rods": "connecting_rod",
    "gears": "gear", "gear_large": "gear", "clutch": "clutch_plate", "a_c_compressor": "air_compressor",
    "ac_compressor": "air_compressor", "engine_valves": "engine_valve", "pistons": "piston",
}


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return re.sub(r"_+", "_", s)


def normalize_engine_bay_name(name: str) -> str | None:
    """Map any spelling of an engine-bay class (or a bare Kaggle index like ``"7"``) onto the canonical
    26-class taxonomy. Returns ``None`` for names that are not part of the taxonomy (caller decides
    whether to drop them)."""
    raw = str(name).strip()
    if raw.isdigit():
        i = int(raw)
        return ENGINE_BAY_CLASSES[i] if 0 <= i < len(ENGINE_BAY_CLASSES) else None
    s = snake(raw)
    s = _ENGINE_BAY_ALIASES.get(s, s)
    return s if s in ENGINE_BAY_CLASSES else None


def engine_bay_class_map(names: list[str], drop_unknown: bool = True) -> dict[str, str | None]:
    """Build a ``remap_classes`` mapping for a dataset whose classes are some spelling of the taxonomy."""
    out: dict[str, str | None] = {}
    for n in names:
        canon = normalize_engine_bay_name(n)
        if canon is None and not drop_unknown:
            canon = snake(n)
        out[n] = canon
    return out


def normalize_parts_catalog_name(name: str) -> str | None:
    """Map any spelling of an isolated-part class ("CARBERATOR", "Cam-Shaft", "Connecting_rod") onto
    :data:`PARTS_CATALOG_CLASSES`; ``None`` when it is not part of the taxonomy."""
    s = snake(name)
    s = _PARTS_CATALOG_ALIASES.get(s, s)
    return s if s in PARTS_CATALOG_CLASSES else None


def parts_catalog_class_map(names: list[str], drop_unknown: bool = True) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        canon = normalize_parts_catalog_name(n)
        if canon is None and not drop_unknown:
            canon = snake(n)
        out[n] = canon
    return out


TAXONOMY_MAPS = {"engine_bay": engine_bay_class_map, "parts_catalog": parts_catalog_class_map}
