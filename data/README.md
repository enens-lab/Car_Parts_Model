# data/

Everything here is rebuildable and git-ignored.

- `raw/<source>/` — downloads exactly as the source ships them (`python scripts/prepare_data.py` fetches them).
- `processed/<recipe>/{train,valid,test}/_annotations.coco.json` + hard-linked images, `dataset_card.json`, `NOTICE.md`.

Never edit files under `processed/` by hand — change the recipe YAML and rebuild.
