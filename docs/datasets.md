# Datasets

All sources are converted into one canonical COCO representation (`carparts.data.coco.CocoDataset`) and then
assembled by *recipes* (`configs/recipes/*.yaml`) into `data/processed/<recipe>/{train,valid,test}/` with a
`dataset_card.json` + `NOTICE.md`. Rebuild everything with `python scripts/prepare_data.py --all`.

## Sources

| Key | What | Images | Labels | Classes | License | Credentials |
|---|---|---|---|---|---|---|
| `carparts_seg` | Exterior body parts (Ultralytics mirror of Roboflow "car-seg" by Gianmarco Russo) | 3,833 (from **585 unique photos**) | polygons | 23 | CC BY 4.0 | none (public GitHub asset, 133 MB) |
| `kaggle_engine_bay` | Engine-bay components (Khaled Chawa) | 1,201 | boxes | 26 | MIT | optional (`kagglehub` fetches public sets anonymously) |
| `rf_engine_bay_parts_stephens` | Engine-bay components re-annotation (Stephens Workspace) | 2,378 | boxes | 52 raw → 26 canonical | MIT | `ROBOFLOW_API_KEY` |
| `rf_car_engine_bay_razeen` | Re-upload of the Kaggle set (off by default; duplicate) | 1,189 | boxes | 26 numeric | CC BY 4.0 | `ROBOFLOW_API_KEY` |
| `rf_engine_parts_detector` | Partial labels — 6 classes only (off by default) | 712 | boxes | 6 | CC BY 4.0 | `ROBOFLOW_API_KEY` |
| `rf_car_under_the_hood` | Partial labels — 3 classes only (off by default) | 120 | boxes | 3 | CC BY 4.0 | `ROBOFLOW_API_KEY` |

Partially-labelled sets are excluded from the default recipes because a detector trained on them learns that
every *unlabelled* component is background. Enable them only for a class subset you actually want.

## The carparts-seg leakage problem

The public split shipped with carparts-seg (3,156 / 401 / 276) is built from Roboflow *augmented copies*: each of
the 585 physical photos appears up to 8 times (`car87_jpg.rf.<md5>.jpg` …), and the copies were split at random:

| Public split | Images | Unique photos | Photos also in train | **Images sharing a photo with train** |
|---|---|---|---|---|
| val | 401 | 295 | 290 (98.3 %) | **392 (97.8 %)** |
| test | 276 | 229 | 223 (97.4 %) | **264 (95.7 %)** |

Any mAP reported on that split is essentially a train-set score. We therefore split **by photo group**
(`carparts.data.naming.group_key` folds the `.rf.<md5>` copies together; `carparts.data.splits.dhash` unions
exact/near duplicates across sources) and freeze the result with a seed:

| `exterior_seg` (grouped, seed 2026) | Images | Annotations |
|---|---|---|
| train | 3,067 | 16,362 |
| valid | 383 | 2,061 |
| test | 383 | 1,951 |
| cross-split photo groups | **0** | |

Consequences: (1) our numbers will be *lower* than anything published on the public split, and they are the honest
ones; (2) the effective dataset is ~585 photos, so expect the model to benefit a lot from more real photos — the
4×GPU box should spend its time on data as much as on model size.

`split.strategy: original` keeps the public split for apples-to-apples comparisons; `classes:` fixes class order;
`class_map:` renames/merges/drops classes per source; `subsample:` (max photo groups) powers the smoke test;
`dedupe_annotations` (default on) removes same-image, same-class boxes with IoU ≥ 0.85.

## The engine-bay corpus is one photo set, labelled twice

Measured with the 64-bit difference hash (`carparts.data.splits.dhash_bits`):

| Comparison | Hamming = 0 | ≤ 2 | ≤ 4 |
|---|---|---|---|
| Roboflow "engine-bay-parts" (2,313 imgs, 640×640) → nearest Kaggle photo | 2,065 (89 %) | 2,311 (99.9 %) | **2,313 (100 %)** |
| Kaggle photo → nearest *other* Kaggle photo | 208 | 267 | 274 |

So the Roboflow set contains **no new photos**: it is a 640×640 re-export of the Kaggle images (its filenames even
keep Kaggle's numbers — 220/220 sampled hash matches agree with the filename lineage), split into two disjoint
sub-labelings (786 images with the numeric Kaggle ids, 836 re-labelled with spelled-out names; no double-labelled
objects). Kaggle itself carries ~270 near-duplicate photos. All of this is folded into photo groups (filename
lineage ∪ Hamming ≤ 4), so nothing leaks — but the corpus is effectively **~1,000 unique engine-bay photos**.

| `engine_bay_det` (grouped, seed 2026) | Images | Annotations | Photo groups |
|---|---|---|---|
| train | 2,810 | 23,441 | 840 |
| valid | 352 | 2,924 | 104 |
| test | 352 | 2,897 | 105 |

Rarest classes (all splits): `atf_oil_reservoir` 3, `oil_filter` 15, `secondary_coolant_reservoir` 24,
`intercooler_coolant_reservoir` 55 — expect near-zero AP there until more photos exist.

`unified_det` (exterior boxes + engine bay): 5,877 / 735 / 735 images, 49 classes, 0 cross-split groups.

## Class taxonomies

**Exterior (23, carparts-seg order):** back_bumper, back_door, back_glass, back_left_door, back_left_light, back_light,
back_right_door, back_right_light, front_bumper, front_door, front_glass, front_left_door, front_left_light,
front_light, front_right_door, front_right_light, hood, left_mirror, object, right_mirror, tailgate, trunk, wheel.
Note the mixed granularity (`front_door` *and* `front_left_door`) and the catch-all `object` (10 train instances) —
`exterior_seg_clean` merges the side variants and drops `object` (13 classes).

**Engine bay (26, Kaggle id order, typos fixed):** inverter_coolant_reservoir, battery, radiator_cap,
windshield_wiper_fluid, fuse_box, power_steering_reservoir, brake_fluid, engine_oil_fill_cap, engine_oil_dipstick,
air_filter_cover, abs_unit, alternator, engine_coolant_reservoir, radiator, air_filter, engine_cover, cold_air_intake,
clutch_fluid_reservoir, transmission_oil_dipstick, intercooler_coolant_reservoir, oil_filter_housing, atf_oil_reservoir,
cabin_air_filter_housing, secondary_coolant_reservoir, electric_motor, oil_filter.
`carparts.constants.normalize_engine_bay_name` maps bare Kaggle ids ("7") and spelling variants
("Oil Filter Housinig", "Engine Oil Dip Stick") onto this list.

## Reports

`artifacts/data_reports/<recipe>_split_report.json` (counts, groups, leakage, per-class per-split instance counts) and
`<recipe>_sources.json` (per-source stats) are regenerated on every build and are meant to be committed.
