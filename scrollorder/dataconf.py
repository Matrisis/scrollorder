"""Configuration of one sample: ``config/<sample>.json``, written by hand from the inventory.

``DataConfig`` has one field per key of the JSON file, in the order of the schema. Nested objects
(``volume``, ``segments``, ``lasagna``, ``volume_9um``) stay plain dicts, exactly as in the file.
Keys that may be left out default to ``None``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PYRAMIDS = ("mean", "stride")
# schema order: the first missing one is the one reported
REQUIRED_KEYS = ("sample", "voxel_mm", "volume", "pyramid", "segments", "tifxyz_to_l0",
                 "recto_offset_px", "mass_from_l0", "overlay_ok")
OPTIONAL_KEYS = ("umbilicus", "lasagna", "volume_9um", "kollesis")


@dataclass(kw_only=True)
class DataConfig:
    sample: str
    voxel_mm: float                      # edge of a level-0 voxel
    volume: dict                         # {"group": key, "levels": {"0": path, ...}, "axes": "zyx"}
    pyramid: str                         # "mean" or "stride": how level n+1 is made from level n
    segments: list[dict]                 # [{"name", "prefix"}], innermost winding first
    tifxyz_to_l0: float                  # factor from tifxyz coordinates to level-0 voxels
    recto_offset_px: float
    mass_from_l0: bool
    overlay_ok: bool | None              # null until someone has looked at the overlay
    umbilicus: str | None = None
    lasagna: dict | None = None          # {"level", "grad_scale", "channels": {name: {"key", "index"}}}
    volume_9um: dict | None = None       # {"pag0": key, "pag50": key}
    kollesis: str | None = None


def load_config(path: str | Path) -> DataConfig:
    """Read and check ``config/<sample>.json``; raises ``ValueError`` naming the first problem."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} : JSON illisible ({exc})") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name} : un objet JSON est attendu")
    for key in REQUIRED_KEYS:
        if key not in doc:
            raise ValueError(f"{key} manquante")
    unknown = sorted(set(doc) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        raise ValueError("clé inconnue : " + ", ".join(unknown))       # a typo must not pass silently
    _check_volume(doc["volume"])
    if doc["pyramid"] not in PYRAMIDS:
        raise ValueError(f'pyramid doit valoir "mean" ou "stride", pas {doc["pyramid"]!r}')
    _check_segments(doc["segments"])
    return DataConfig(**doc)


def _check_volume(volume) -> None:
    if not isinstance(volume, dict):
        raise ValueError("volume doit être un objet")
    for key in ("group", "levels"):
        if key not in volume:
            raise ValueError(f"volume.{key} manquante")
    levels = volume["levels"]
    if not isinstance(levels, dict) or not levels or not all(str(k).isdigit() for k in levels):
        raise ValueError('volume.levels doit associer un numéro de niveau à un chemin, par exemple {"0": "0"}')


def _check_segments(segments) -> None:
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments doit être une liste non vide")
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{i}] doit être un objet")
        for key in ("name", "prefix"):
            if key not in segment:
                raise ValueError(f"segments[{i}].{key} manquante")
