"""Dataset sources: each knows how to download itself and how to become a :class:`CocoDataset`.

The registry is the single place that records **license and attribution** for every pixel that can
enter a training set — ``scripts/prepare_data.py`` copies it into the processed dataset card.
"""
from __future__ import annotations

from .base import Source, SourceInfo, registry
from .carparts_seg import CarpartsSegSource
from .kaggle_engine_bay import KaggleEngineBaySource
from .roboflow import RoboflowSource

# ------------------------------------------------------------------------------ registrations
registry.register(CarpartsSegSource())
registry.register(KaggleEngineBaySource())

# Roboflow Universe extras. Only the fully-labelled 26-class re-labelling is enabled by default;
# partially-labelled sets (3-6 classes on engine-bay photos) would teach the model that every
# unlabelled component is background, so they stay opt-in and are documented in docs/datasets.md.
registry.register(RoboflowSource(
    name="rf_engine_bay_parts_stephens",
    workspace="stephens-workspace-gjyrj", project="engine-bay-parts", version=1,
    info=SourceInfo(
        title="engine-bay-parts (Stephens Workspace)", license="MIT",
        url="https://universe.roboflow.com/stephens-workspace-gjyrj/engine-bay-parts",
        attribution="Stephens Workspace, 'engine-bay-parts Dataset', Roboflow Universe, 2026 (MIT).",
        notes="2,378 engine-bay photos; 52 raw classes = the 26 Kaggle numeric ids + the same 26 spelled out. "
              "Normalised onto the canonical 26-class taxonomy by `class_map: engine_bay`.",
    ),
    taxonomy="engine_bay",
))
registry.register(RoboflowSource(
    name="rf_car_engine_bay_razeen",
    workspace="razeenxs-workplace", project="car-engine-bay", version=1,
    info=SourceInfo(
        title="Car Engine Bay (Razeenxs workplace)", license="CC-BY-4.0",
        url="https://universe.roboflow.com/razeenxs-workplace/car-engine-bay",
        attribution="Razeenxs workplace, 'Car Engine Bay Dataset', Roboflow Universe, Dec 2024 (CC BY 4.0).",
        notes="1,189 photos with the 26 numeric Kaggle ids — almost certainly a re-upload of the Kaggle set. "
              "Useful as a fallback when Kaggle credentials are unavailable; duplicates are folded by dhash.",
    ),
    taxonomy="engine_bay",
))
registry.register(RoboflowSource(
    name="rf_engine_parts_detector",
    workspace="final-year-project-5vbtb", project="engine-parts-detector", version=3,
    info=SourceInfo(
        title="Engine Parts Detector (Final Year Project)", license="CC-BY-4.0",
        url="https://universe.roboflow.com/final-year-project-5vbtb/engine-parts-detector",
        attribution="Final Year Project, 'Engine Parts Detector Dataset', Roboflow Universe, Dec 2022 (CC BY 4.0).",
        notes="712 photos, 6 classes only (Battery, Air Coolant, Fuse-Box, Main Engine, Radiator, Windscreen Washer). "
              "PARTIAL labels — opt-in only.",
    ),
    taxonomy="engine_bay",
))
registry.register(RoboflowSource(
    name="rf_car_under_the_hood",
    workspace="projectcar", project="car-under-the-hood", version=2,
    info=SourceInfo(
        title="car under the hood (projectcar)", license="CC-BY-4.0",
        url="https://universe.roboflow.com/projectcar/car-under-the-hood",
        attribution="projectcar, 'car under the hood Dataset', Roboflow Universe, Jan 2023 (CC BY 4.0).",
        notes="120 photos, 3 classes (battery, airfilter, engine). PARTIAL labels — opt-in only.",
    ),
    taxonomy="engine_bay",
))

# Isolated-part ("parts catalog") photos — the data behind powertrain component recognition.
registry.register(RoboflowSource(
    name="rf_used_auto_parts_50",
    workspace="used-auto-parts-classification", project="car-parts-ybiev-whary", version=2,
    info=SourceInfo(
        title="car parts (Used auto parts classification) — 50 isolated-part classes", license="Public Domain",
        url="https://universe.roboflow.com/used-auto-parts-classification/car-parts-ybiev-whary",
        attribution="Used auto parts classification, 'car parts Dataset', Roboflow Universe, 2024 (Public Domain).",
        notes="8,694 catalog-style photos, one part per image, 50 classes incl. engine block, cylinder head, "
              "camshaft, crankshaft, piston, engine valve, valve lifter, torque converter, clutch/pressure plate, "
              "transmission. Same taxonomy as Kaggle gpiosenka/car-parts-40-classes (Apache 2.0).",
    ),
    taxonomy="parts_catalog",
))
registry.register(RoboflowSource(
    name="rf_engine_internals_383",
    workspace="engineparts", project="engine-parts-c2xfb", version=1,
    info=SourceInfo(
        title="Engine Parts (engineparts) — cam/crank/rods/pistons/heads", license="CC-BY-4.0",
        url="https://universe.roboflow.com/engineparts/engine-parts-c2xfb",
        attribution="engineparts, 'Engine Parts Dataset', Roboflow Universe, Oct 2024 (CC BY 4.0).",
        notes="383 photos, 5 classes (Cam-Shaft, Connecting_rod, Piston, Crank-shaft, Cylinder-head); v1 export is "
              "3x augmented (1,071 images) — copies are folded by the photo grouping.",
    ),
    taxonomy="parts_catalog",
))
registry.register(RoboflowSource(
    name="rf_engine_internals_129",
    workspace="project-tevws", project="engine-parts-iejh6", version=1,
    info=SourceInfo(
        title="engine parts (project-tevws) — camshaft/piston/connecting rod/gear", license="CC-BY-4.0",
        url="https://universe.roboflow.com/project-tevws/engine-parts-iejh6",
        attribution="project-tevws, 'engine parts Dataset', Roboflow Universe, Jul 2023 (CC BY 4.0).",
        notes="129 photos, 4 classes; v1 export is 3x augmented (309 images).",
    ),
    taxonomy="parts_catalog",
))
# NOT registered: car-parts-jv0or/car-parts-detection-owvwe (10 classes incl. crankshaft/clutch plate) — license
# field is 'Private'; clutch/clutch-detecton — defect Good/Bad labels, not part identity.

__all__ = ["Source", "SourceInfo", "registry"]
