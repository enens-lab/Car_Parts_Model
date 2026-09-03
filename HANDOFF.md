# HANDOFF — car-parts model

_Mission log. Newest state at the top. Owner of the next phase: whoever runs the 4×RTX 6000 Ada box._

## State (2026-09-03)

**Done on the laptop (RTX 4070 Laptop 8 GB, 16 GB RAM, Windows 11):**

- Full pipeline implemented and unit-tested (`pytest tests`): sources → canonical COCO → leakage-safe recipes →
  RF-DETR fine-tuning → per-class COCO eval → predict → ONNX export → FastAPI demo.
- `scripts/smoke_test.py` **passes all four stages** (prepare 10 s, train+eval 30 s, predict 0.5 s, ONNX export 5 s).
- Datasets downloaded and processed (train / valid / test images): `exterior_seg` 3,067 / 383 / 383 (23 classes),
  `exterior_seg_clean` 3,067 / 383 / 383 (13), `engine_bay_det` 2,810 / 352 / 352 (26), `unified_det`
  5,877 / 735 / 735 (49) — all with **0 cross-split photo groups**. Reports in `artifacts/data_reports/`.
- Proof-of-concept model: `poc_exterior_seg_nano` (rfdetr-seg-nano @312, 12 epochs) — see scoreboard.

**Scoreboard** (grouped split, test = 383 exterior images; mask AP unless noted)

| Run | Model | Data | Epochs | test bbox AP / AP50 | test mask AP / AP50 | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|
| `smoke` | rfdetr-seg-nano @312 | 24-photo subset | 1 | — | — | 0.5 min | pipeline check only |
| `poc_exterior_seg_nano` | rfdetr-seg-nano @312 | exterior_seg | 12 | _fill from artifacts/runs/poc_exterior_seg_nano/summary.json_ | | | laptop POC |

Production targets (big box): `exterior_seg` with rfdetr-seg-xlarge @624 and `engine_bay_det` with rfdetr-large @704,
100 epochs, effective batch 32.

## Machine notes

**Laptop (this box).** 16 GB RAM is the bottleneck, not the GPU. Two things bit us:
1. numpy's OpenBLAS aborts (`Memory allocation still failed after 10 retries`) unless BLAS threads are capped —
   `import carparts` sets `OPENBLAS_NUM_THREADS=4` before numpy loads; always import it first (scripts do).
2. A `torch.save` of `last.ckpt` died with `MemoryError` when a dataset build and pytest ran concurrently with
   training. Run training **alone** on this machine; keep `num_workers: 2`.

**4×RTX 6000 Ada box (Linux, ~250 GB RAM).** `bash scripts/remote_setup.sh` installs uv + the environment, pulls the
data and builds every recipe. Multi-GPU is Lightning DDP via `torchrun` (`configs/train/rfdetr_4x48gb.yaml` sets
`devices: auto`). RF-DETR downloads pretrained weights to `~/.roboflow/models/` on first use — vendor them if the box
is offline (`pretrain_weights=` accepts a local path).

## Next missions (in order)

```bash
# 0. sanity
uv run python scripts/smoke_test.py

# 1. exterior segmentation, production model
uv run torchrun --nproc_per_node=4 scripts/train.py --recipe exterior_seg --config rfdetr_4x48gb --name exterior_seg_xl
uv run python scripts/evaluate.py --run exterior_seg_xl            # per-class table -> artifacts/runs/exterior_seg_xl/eval_test.json

# 2. engine-bay detection, production model
uv run torchrun --nproc_per_node=4 scripts/train.py --recipe engine_bay_det --config rfdetr_4x48gb --model rfdetr-large --name engine_bay_l

# 3. the side-agnostic exterior taxonomy (product decision pending: 23 vs 13 classes)
uv run torchrun --nproc_per_node=4 scripts/train.py --recipe exterior_seg_clean --config rfdetr_4x48gb --name exterior_clean_xl

# 4. optional AGPL benchmark for the model-selection memo (never ship its outputs)
uv sync --extra baseline-agpl && uv run python scripts/baseline_ultralytics.py --recipe exterior_seg --model yolo26m-seg.pt --epochs 100 --i-accept-agpl

# 5. export + demo
uv run python scripts/export.py --run exterior_seg_xl --format onnx
uv run python app/server.py --run exterior_seg_xl --run engine_bay_l --host 0.0.0.0
```

Then commit `artifacts/runs/<run>/summary.json` + `eval_test.json` and add a scoreboard row here.

## Ground rules

1. **The grouped split is frozen** (`seed: 2026` in every recipe). Never report numbers from the public
   carparts-seg split — it leaks 96–98 % of its eval images into train (`docs/datasets.md`).
2. Test split is touched once per run, by `evaluate_checkpoint`. Model selection uses `valid/`.
3. Only models in `MODEL_REGISTRY` ship (Apache-2.0). No `rfdetr[plus]` XL/2XL detection weights, no Ultralytics.
4. Every processed dataset's `NOTICE.md` travels with anything trained on it.
5. Data changes go through recipes + `prepare_data.py`; never hand-edit `data/processed/`.
6. Commit `summary.json`/`eval_*.json`/data reports; never weights, raw data, or `.env`.

## Known issues & ideas

- Kaggle engine-bay photos contain ~270 near-duplicates; the Roboflow "engine-bay-parts" set is a 640×640 re-export
  of the *same* photos (two disjoint sub-labelings: 786 images with the numeric Kaggle ids, 836 re-labelled by name).
  Both are folded into one photo group per source photo, so no leakage — but the engine-bay corpus is really
  ~1,200 photos. More real engine-bay photos are the highest-value next data investment.
- `object` (10 instances) and `tailgate` (88) in carparts-seg are too rare to learn; `exterior_seg_clean` drops
  `object`. Consider `min_instances_per_class` once the production baseline is in.
- Ideas ported from the knee-MRI project once baselines exist: multi-seed rank ensembles, long-schedule check,
  pseudo-labelling unlabelled engine-bay photos with the best model (noisy student), SAM-2 pseudo-masks for
  engine-bay boxes to unlock a unified segmentation model.
