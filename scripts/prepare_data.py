"""Download the sources and build processed datasets from recipes.

Examples
--------
  python scripts/prepare_data.py --list
  python scripts/prepare_data.py --recipe exterior_seg --recipe engine_bay_det
  python scripts/prepare_data.py --all --no-download          # rebuild from what is already on disk
  python scripts/prepare_data.py --recipe engine_bay_det --strict   # fail instead of skipping a source
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import sys
from pathlib import Path

from carparts.config import CONFIG_DIR, load_paths, load_recipe
from carparts.data.recipe import build_recipe
from carparts.sources import registry


def list_everything() -> None:
    print("Sources (license / credentials):")
    for s in registry:
        env = ", ".join(s.info.requires_env) or "none"
        print(f"  {s.name:32s} {s.info.license:10s} needs: {env:20s} {s.info.title}")
    print("\nRecipes:")
    for p in sorted((CONFIG_DIR / "recipes").glob("*.yaml")):
        r = load_recipe(p.stem)
        srcs = [e["name"] if isinstance(e, dict) else e for e in r["sources"]]
        print(f"  {r['name']:22s} {r['task']:13s} split={r['split']['strategy']:8s} sources={srcs}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", action="append", default=[], help="recipe name (repeatable)")
    ap.add_argument("--all", action="store_true", help="build every recipe in configs/recipes")
    ap.add_argument("--no-download", action="store_true", help="never hit the network; use raw data on disk")
    ap.add_argument("--strict", action="store_true", help="fail when any source cannot be loaded")
    ap.add_argument("--list", action="store_true", help="list sources and recipes, then exit")
    args = ap.parse_args(argv)

    if args.list:
        list_everything()
        return 0
    names = args.recipe or ([p.stem for p in sorted((CONFIG_DIR / "recipes").glob("*.yaml"))] if args.all else [])
    if not names:
        ap.error("give --recipe NAME (repeatable), --all, or --list")

    paths = load_paths().ensure()
    failures = 0
    for name in names:
        recipe = load_recipe(name)
        try:
            build_recipe(recipe, paths.raw, paths.processed, paths.reports, download=not args.no_download,
                         strict=args.strict)
        except Exception as e:
            failures += 1
            print(f"[prepare_data] recipe {name!r} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            if args.strict:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
