from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SPLIT_ORDER = ("train", "dev", "test", "external_final")


def stable_bucket(image_id: int, salt: str) -> int:
    """Return a deterministic 0..999 bucket without depending on Python hash()."""

    payload = f"{salt}:{int(image_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1000


def assign_split(
    image_id: int,
    *,
    salt: str,
    train_cut: int = 700,
    dev_cut: int = 850,
) -> str:
    if not 0 < train_cut < dev_cut < 1000:
        raise ValueError("split cuts must satisfy 0 < train_cut < dev_cut < 1000")
    bucket = stable_bucket(image_id, salt)
    if bucket < train_cut:
        return "train"
    if bucket < dev_cut:
        return "dev"
    return "test"


def _safe_relative_image(root: Path, file_name: str) -> str:
    # COCO file names are simple, but reject traversal before any file is opened.
    rel = Path(str(file_name).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe image file name: {file_name!r}")
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath((str(candidate), str(root_resolved))) != str(root_resolved):
        raise ValueError(f"image escapes root: {file_name!r}")
    return rel.as_posix()


def _normalise_caption(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError("caption must not be empty")
    return text


def build_coco_manifest(
    annotation_json: Path,
    image_root: Path,
    output_json: Path,
    *,
    split_salt: str = "mosaic-coco5k-v1",
    train_cut: int = 700,
    dev_cut: int = 850,
    require_files: bool = True,
) -> dict[str, Any]:
    """Build an image-cluster split from official COCO captions JSON.

    All captions belonging to one image are kept in one split. This prevents a
    common but serious leakage where four captions of an image are in train and
    the fifth caption is used as test.
    """

    annotation_json = Path(annotation_json).resolve()
    image_root = Path(image_root).resolve()
    output_json = Path(output_json).resolve()
    payload = json.loads(annotation_json.read_text(encoding="utf-8"))
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO captions JSON must contain images and annotations lists")

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for row in annotations:
        if not isinstance(row, Mapping):
            raise ValueError("caption annotation must be an object")
        image_id = int(row["image_id"])
        captions_by_image[image_id].append(_normalise_caption(row["caption"]))

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in sorted(images, key=lambda item: int(item["id"])):
        image_id = int(row["id"])
        if image_id in seen:
            raise ValueError(f"duplicate COCO image id {image_id}")
        seen.add(image_id)
        captions = captions_by_image.get(image_id, [])
        if not captions:
            raise ValueError(f"image {image_id} has no captions")
        # COCO normally has five; preserve all while recording the observed count.
        rel_name = _safe_relative_image(image_root, str(row["file_name"]))
        absolute = image_root / Path(rel_name)
        if require_files and not absolute.is_file():
            raise FileNotFoundError(absolute)
        records.append(
            {
                "image_id": image_id,
                "file_name": rel_name,
                "captions": captions,
                "split": assign_split(
                    image_id,
                    salt=split_salt,
                    train_cut=train_cut,
                    dev_cut=dev_cut,
                ),
            }
        )

    if not records:
        raise ValueError("COCO manifest is empty")
    counts = {split: sum(row["split"] == split for row in records) for split in SPLIT_ORDER}
    manifest: dict[str, Any] = {
        "schema_version": "mosaic.manifest.v1",
        "dataset": "COCO-2017-val-captions",
        "annotation_file": str(annotation_json),
        "image_root": str(image_root),
        "split": {
            "method": "sha256(image_id + salt) modulo 1000",
            "salt": split_salt,
            "train_cut": train_cut,
            "dev_cut": dev_cut,
            "counts": counts,
        },
        "images": records,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path, *, check_digest: bool = True) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "mosaic.manifest.v1":
        raise ValueError("unsupported manifest schema")
    if check_digest and manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("manifest_sha256 does not match manifest contents")
    records = manifest.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest images must be a non-empty list")
    ids = [int(row["image_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate image ids")
    for row in records:
        if row.get("split") not in SPLIT_ORDER:
            raise ValueError("manifest contains an invalid split")
        if not isinstance(row.get("captions"), list) or not row["captions"]:
            raise ValueError("every image must have at least one caption")
    return manifest


def iter_images(manifest: Mapping[str, Any], split: str | None = None) -> Iterator[dict[str, Any]]:
    if split is not None and split not in SPLIT_ORDER:
        raise ValueError(f"invalid split: {split}")
    for row in manifest["images"]:
        if split is None or row["split"] == split:
            yield row


def caption_pairs(
    manifest: Mapping[str, Any], split: str, *, max_images: int | None = None
) -> list[dict[str, Any]]:
    rows = list(iter_images(manifest, split))
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        rows = rows[:max_images]
    pairs: list[dict[str, Any]] = []
    for row in rows:
        for caption_index, caption in enumerate(row["captions"]):
            pairs.append(
                {
                    "image_id": int(row["image_id"]),
                    "file_name": str(row["file_name"]),
                    "caption_index": int(caption_index),
                    "caption": str(caption),
                    "split": split,
                }
            )
    return pairs


def build_toy_manifest(output_json: Path, *, image_root: Path) -> dict[str, Any]:
    """Create a deterministic manifest used only by unit tests and smoke runs."""

    records = [
        {
            "image_id": index,
            "file_name": f"toy_{index:03d}.png",
            "captions": [f"a synthetic object number {index}", f"toy example {index}"],
            "split": ("train", "dev", "test")[index // 4],
        }
        for index in range(12)
    ]
    manifest: dict[str, Any] = {
        "schema_version": "mosaic.manifest.v1",
        "dataset": "MOSAIC-toy-not-for-evaluation",
        "annotation_file": "synthetic",
        "image_root": str(Path(image_root).resolve()),
        "split": {"method": "fixed toy split", "salt": "toy", "counts": {"train": 4, "dev": 4, "test": 4}},
        "images": records,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "SPLIT_ORDER",
    "assign_split",
    "build_coco_manifest",
    "build_toy_manifest",
    "caption_pairs",
    "iter_images",
    "load_manifest",
    "manifest_sha256",
    "stable_bucket",
]
