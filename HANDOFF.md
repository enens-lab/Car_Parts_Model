# HANDOFF — car-parts model

_Mission log for whoever runs the GPU box next. Newest state at the top._

## GPU policy on the training box (read first)

The box has 4× RTX 6000 Ada (48 GB each) and ~250 GB RAM, shared with other workloads.

| GPU id | Status | Use |
|---|---|---|
| **0** | **FORBIDDEN** | never touch — not even for a smoke test or a `predict` |
| 1, 2 | borrowable | only when their other workloads are idle (check `nvidia-smi` first); release when done |
| **3** | **default** | all carparts work runs here unless you deliberately borrow 1–2 |

Mechanics:

- `scripts/remote_setup.sh` exports `CUDA_VISIBLE_DEVICES=3`; put the same line in your shell profile on the box.
- `import carparts` applies the policy (`src/carparts/gpu_policy.py`): on a machine with ≥ 4 GPUs it defaults
  `CUDA_VISIBLE_DEVICES=3` when unset and **aborts** if the visible set includes GPU 0
  (`CARPARTS_ALLOW_GPU0=1` overrides — do not set it on this box).
- Inside a process, the visible GPU is always `cuda:0` (PyTorch renumbers) — that is fine and expected.
- Multi-GPU is opt-in only: `CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 …` with `rfdetr_3x48gb.yaml`.

## State (2026-09-03)

**Done on the laptop (RTX 4070 Laptop 8 GB, 16 GB RAM, Windows 11):**

- Full pipeline implemented and unit-tested (`pytest tests`, 23 tests): sources → canonical COCO → leakage-safe
  recipes → RF-DETR fine-tuning → per-class COCO eval → predict → ONNX export → FastAPI demo.
- `scripts/smoke_test.py` passes all four stages (prepare 10 s, train+eval 30 s, predict 0.5 s, ONNX export 5 s).
- Datasets downloaded and processed (train / valid / test images): `exterior_seg` 3,067 / 383 / 383 (23 classes),
  `exterior_seg_clean` 3,067 / 383 / 383 (13), `engine_bay_det` 2,810 / 352 / 352 (26), `unified_det`
  5,877 / 735 / 735 (49) — all with **0 cross-split photo groups**. Reports in `artifacts/data_reports/`.
- Proof-of-concept `poc_exterior_seg_nano` (rfdetr-seg-nano @312, 12 epochs, in-process data loading) — see scoreboard.

**Scoreboard** (frozen grouped split, seed 2026; test = 383 exterior photos)

| Run | Model | Data | Epochs | val mask mAP (best) | test bbox AP / AP50 | test mask AP / AP50 | Wall-clock | Where |
|---|---|---|---|---|---|---|---|---|
| `smoke` | rfdetr-seg-nano @312 | 24-photo subset | 1 | 0.01 | — | — | 0.5 min | laptop, pipeline check only |
| `poc_exterior_seg_nano` | rfdetr-seg-nano @312 | exterior_seg | 12 | 0.488 @ epoch 8 (run in progress) | _see artifacts/runs/poc_exterior_seg_nano/summary.json_ | | ~65 min | laptop POC |

Production targets (box, GPU 3): `exterior_seg` with rfdetr-seg-xlarge @624 and `engine_bay_det` with rfdetr-large
@704, 100 epochs, effective batch 32 — `configs/train/rfdetr_48gb.yaml`.

## Machine notes

**Laptop.** 16 GB RAM is the bottleneck, not the GPU:
1. numpy's OpenBLAS aborts (`Memory allocation still failed after 10 retries`) unless BLAS threads are capped —
   `import carparts` sets `OPENBLAS_NUM_THREADS=4` before numpy loads; every script imports it first.
2. `torch.save` of `last.ckpt` died with `MemoryError` when a dataset build and pytest ran concurrently with
   training, and spawned Windows DataLoader workers (~1.5 GB each) hung a run. Run training **alone** here with
   `num_workers: 0` (`poc_8gb.yaml`); `rfdetr_8gb.yaml` uses 2 workers and needs Chrome closed.

**Box (Linux, 4× RTX 6000 Ada, ~250 GB RAM).** `bash scripts/remote_setup.sh` installs uv + the environment, pulls
the data, builds every recipe and prints the GPU it will use. RF-DETR downloads pretrained weights to
`~/.roboflow/models/` on first use — vendor them if the box is offline (`pretrain_weights=` takes a local path).
A single RTX 6000 Ada trains seg-xlarge @624 at batch 8 comfortably; the 48 GB config uses `grad_accum_steps: 4`
to reach effective batch 32 on one GPU.

## Next missions (in order, all on GPU 3)

```bash
export CUDA_VISIBLE_DEVICES=3          # remote_setup.sh does this too; the code refuses GPU 0 regardless

# 0. sanity
uv run python scripts/smoke_test.py

# 1. exterior segmentation, production model  (~1 GPU-day)
uv run python scripts/train.py --recipe exterior_seg --config rfdetr_48gb --name exterior_seg_xl
uv run python scripts/evaluate.py --run exterior_seg_xl              # per-class table -> artifacts/runs/exterior_seg_xl/eval_test.json

# 2. engine-bay detection, production model
uv run python scripts/train.py --recipe engine_bay_det --config rfdetr_48gb --model rfdetr-large --name engine_bay_l

# 3. side-agnostic exterior taxonomy (product decision pending: 23 vs 13 classes)
uv run python scripts/train.py --recipe exterior_seg_clean --config rfdetr_48gb --name exterior_clean_xl

# 4. optional AGPL benchmark for the model-selection memo (never ship its outputs)
uv sync --extra baseline-agpl && uv run python scripts/baseline_ultralytics.py --recipe exterior_seg --model yolo26m-seg.pt --epochs 100 --i-accept-agpl

# 5. export + demo (bind to 0.0.0.0 only if the box's firewall allows it)
uv run python scripts/export.py --run exterior_seg_xl --format onnx
uv run python app/server.py --run exterior_seg_xl --run engine_bay_l --host 0.0.0.0 --port 8000
```

Only if GPUs 1 and 2 are idle and you have the go-ahead to borrow them:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 uv run torchrun --nproc_per_node=3 scripts/train.py --recipe exterior_seg --config rfdetr_3x48gb --name exterior_seg_xl_3gpu
```

After each run: commit `artifacts/runs/<run>/summary.json` + `eval_test.json`, add a scoreboard row here.

## Ground rules

1. **GPU 0 is forbidden. GPU 3 is the default.** Borrow 1–2 only when idle; never leave a job on them unattended.
2. **The grouped split is frozen** (`seed: 2026` in every recipe). Never report numbers from the public
   carparts-seg split — it leaks 96–98 % of its eval images into train (`docs/datasets.md`).
3. Test split is touched once per run, by `evaluate_checkpoint`. Model selection uses `valid/`.
4. Only models in `MODEL_REGISTRY` ship (Apache-2.0). No `rfdetr[plus]` XL/2XL detection weights, no Ultralytics.
5. Every processed dataset's `NOTICE.md` travels with anything trained on it.
6. Data changes go through recipes + `prepare_data.py`; never hand-edit `data/processed/`.
7. Commit `summary.json` / `eval_*.json` / data reports; never weights, raw data, or `.env`.
8. No detached watcher processes on the box; long runs go in `tmux`/`screen` with logs under `artifacts/logs/`.

## Known issues & ideas

- Kaggle engine-bay photos contain ~270 near-duplicates; the Roboflow "engine-bay-parts" set is a 640×640
  re-export of the *same* photos (two disjoint sub-labelings: 786 images with the numeric Kaggle ids, 836
  re-labelled by name). Both fold into one photo group per source photo, so no leakage — but the engine-bay corpus
  is really ~1,000 unique photos. More real engine-bay photos are the highest-value next data investment.
- `object` (10 instances) and `tailgate` (88) in carparts-seg are too rare to learn; `exterior_seg_clean` drops
  `object`. Consider `min_instances_per_class` once the production baseline is in.
- Rarest engine-bay classes (`atf_oil_reservoir` 3, `oil_filter` 15, `secondary_coolant_reservoir` 24) will
  score ~0 AP until more photos exist.
- Ideas ported from the knee-MRI project once baselines exist: multi-seed rank ensembles, long-schedule check,
  pseudo-labelling unlabelled engine-bay photos with the best model (noisy student), SAM-2 pseudo-masks for
  engine-bay boxes to unlock a unified segmentation model.
