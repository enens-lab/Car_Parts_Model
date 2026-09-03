"""File-name conventions shared by the loaders."""
from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{8,}$", re.IGNORECASE)
_EXT_TOKENS = re.compile(r"(?:[_.-](?:jpe?g|png|bmp|webp))+$", re.IGNORECASE)


def group_key(file_name: str) -> str:
    """Physical-photo identity for leakage-safe splitting.

    Roboflow exports name augmented copies ``<stem>_jpg.rf.<md5>.jpg``; all copies of one photo must end up
    on the same side of the train/test line. ``car87_jpg.rf.27fc37a7....jpg`` -> ``car87``;
    ``new_10_png_jpg.rf.<md5>.jpg`` -> ``new_10``; anything else -> its bare stem.
    """
    stem = Path(file_name).stem
    stem = _RF_SUFFIX.sub("", stem)
    stem = _EXT_TOKENS.sub("", stem)
    return stem or Path(file_name).stem
