from carparts.constants import (ENGINE_BAY_CLASSES, PARTS_CATALOG_CLASSES, POWERTRAIN_ACCESSORY_CLASSES,
                                POWERTRAIN_CORE_CLASSES, engine_bay_class_map, normalize_engine_bay_name,
                                normalize_parts_catalog_name, parts_catalog_class_map)


def test_engine_bay_normalisation():
    assert normalize_engine_bay_name("7") == "engine_oil_fill_cap"
    assert normalize_engine_bay_name("Oil Filter Housinig") == "oil_filter_housing"
    assert normalize_engine_bay_name("Engine Oil Dip Stick") == "engine_oil_dipstick"
    assert normalize_engine_bay_name("Fuse-Box") == "fuse_box"
    assert normalize_engine_bay_name("engine-bay-parts") is None
    assert len(ENGINE_BAY_CLASSES) == 26 and len(set(ENGINE_BAY_CLASSES)) == 26
    m = engine_bay_class_map(["0", "Battery", "unknown thing"])
    assert m == {"0": "inverter_coolant_reservoir", "Battery": "battery", "unknown thing": None}


def test_parts_catalog_normalisation():
    assert normalize_parts_catalog_name("CARBERATOR") == "carburetor"
    assert normalize_parts_catalog_name("Cam-Shaft") == "camshaft"
    assert normalize_parts_catalog_name("Connecting_rod") == "connecting_rod"
    assert normalize_parts_catalog_name("TORQUE CONVERTER") == "torque_converter"
    assert normalize_parts_catalog_name("HEADLIGHTS") == "headlight"
    assert normalize_parts_catalog_name("gear") == "gear"
    assert normalize_parts_catalog_name("car parts") is None  # Roboflow parent category
    assert len(PARTS_CATALOG_CLASSES) == len(set(PARTS_CATALOG_CLASSES)) == 52
    assert set(POWERTRAIN_CORE_CLASSES) <= set(PARTS_CATALOG_CLASSES)
    assert set(POWERTRAIN_ACCESSORY_CLASSES) <= set(PARTS_CATALOG_CLASSES)
    assert parts_catalog_class_map(["PISTON", "Piston", "pistons"]) == {"PISTON": "piston", "Piston": "piston",
                                                                         "pistons": "piston"}
