from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.coco import CocoDataset


@dataclass
class SourceInfo:
    title: str
    license: str                 # SPDX-ish: CC-BY-4.0, MIT, ...
    url: str
    attribution: str             # the exact line we ship in NOTICE / dataset cards
    notes: str = ""
    requires_env: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Source(ABC):
    """A downloadable, convertible dataset."""

    name: str
    info: SourceInfo

    # ---------------------------------------------------------------- hooks
    def raw_dir(self, raw_root: Path) -> Path:
        return raw_root / self.name

    def is_downloaded(self, raw_root: Path) -> bool:
        return self.raw_dir(raw_root).exists() and any(self.raw_dir(raw_root).iterdir())

    def missing_env(self) -> list[str]:
        return [k for k in self.info.requires_env if not os.environ.get(k)]

    @abstractmethod
    def download(self, raw_root: Path) -> Path:
        """Fetch into ``raw_dir``; idempotent."""

    @abstractmethod
    def load(self, raw_root: Path) -> dict[str, CocoDataset]:
        """Return ``{split_name: CocoDataset}`` using the *source's own* splits when it has them,
        otherwise ``{"all": ds}``. Class names are the source's canonical names."""


class Registry:
    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}

    def register(self, s: Source, replace: bool = False) -> None:
        if s.name in self._sources and not replace:
            raise KeyError(f"source {s.name!r} already registered")
        self._sources[s.name] = s

    def get(self, name: str) -> Source:
        try:
            return self._sources[name]
        except KeyError:
            raise KeyError(f"unknown source {name!r}; known: {sorted(self._sources)}") from None

    def names(self) -> list[str]:
        return sorted(self._sources)

    def __iter__(self):
        return iter(self._sources.values())


registry = Registry()
