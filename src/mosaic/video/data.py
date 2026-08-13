from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "mosaic.video_manifest.v1"


def _normalise_caption(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError("video caption must not be empty")
    return text


def _digest(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _selection_key(video_id: str, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}:{video_id}".encode()).digest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_msrvtt_manifests(
    train_json: Path,
    test_json: Path,
    video_root: Path,
    train_output: Path,
    test_output: Path,
    *,
    dev_count: int = 1000,
    dev_salt: str = "mosaic-msrvtt-dev-v1",
    require_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_rows = json.loads(Path(train_json).read_text(encoding="utf-8"))
    test_rows = json.loads(Path(test_json).read_text(encoding="utf-8"))
    if not isinstance(train_rows, list) or not isinstance(test_rows, list):
        raise ValueError("MSR-VTT split JSON files must be lists")
    train_ids = {str(row["video_id"]) for row in train_rows}
    test_ids = {str(row["video_id"]) for row in test_rows}
    if len(train_ids) != len(train_rows) or len(test_ids) != len(test_rows):
        raise ValueError("MSR-VTT rows must contain one object per unique video")
    if train_ids & test_ids:
        raise ValueError("MSR-VTT train/test video ids overlap")
    if not 0 < int(dev_count) < len(train_rows):
        raise ValueError("dev_count must leave non-empty train data")
    dev_ids = set(sorted(train_ids, key=lambda value: _selection_key(value, dev_salt))[: int(dev_count)])
    root = Path(video_root).resolve()
    numeric_ids: set[int] = set()
    file_names: set[str] = set()

    def record(row: Mapping[str, Any], split: str) -> dict[str, Any]:
        video_id = str(row["video_id"])
        file_name = Path(str(row["video"])).name
        numeric_id = int(row["id"])
        category = int(row["category"])
        if not video_id or not file_name.lower().endswith(".mp4"):
            raise ValueError("video id and MP4 filename must be non-empty")
        if numeric_id in numeric_ids or file_name in file_names:
            raise ValueError("numeric video ids and filenames must be globally unique")
        if not 0 <= category < 20:
            raise ValueError("MSR-VTT category must be in [0, 20)")
        numeric_ids.add(numeric_id)
        file_names.add(file_name)
        path = root / file_name
        if require_files and not path.is_file():
            raise FileNotFoundError(path)
        raw_captions = row["caption"]
        captions = (
            [_normalise_caption(value) for value in raw_captions]
            if isinstance(raw_captions, list)
            else [_normalise_caption(raw_captions)]
        )
        return {
            "video_id": video_id,
            "numeric_id": numeric_id,
            "file_name": file_name,
            "captions": captions,
            "category": category,
            "split": split,
        }

    train_records = [
        record(row, "dev" if str(row["video_id"]) in dev_ids else "train")
        for row in sorted(train_rows, key=lambda item: int(item["id"]))
    ]
    test_records = [record(row, "test") for row in sorted(test_rows, key=lambda item: int(item["id"]))]
    if any(len(row["captions"]) != 1 for row in test_records):
        raise ValueError("JSFusion 1K-A test requires exactly one query caption per video")
    common = {
        "schema_version": SCHEMA,
        "dataset": "MSR-VTT-1K-A",
        "mirror": "friedrichor/MSR-VTT@c1af215a96934854f42683c19c51391aaee6f962",
        "video_root": str(root),
        "protocol": "JSFusion 1K-A",
        "source_files": {
            "train_json_sha256": _sha256_file(train_json),
            "test_json_sha256": _sha256_file(test_json),
        },
    }
    train_manifest: dict[str, Any] = {
        **common,
        "selection": {
            "dev_method": "lowest sha256(salt:video_id)",
            "dev_salt": dev_salt,
            "counts": {"train": len(train_rows) - dev_count, "dev": dev_count},
        },
        "videos": train_records,
    }
    test_manifest: dict[str, Any] = {
        **common,
        "selection": {
            "test_protocol": "official 1K-A one query caption per video",
            "counts": {"test": len(test_rows)},
        },
        "videos": test_records,
    }
    train_manifest["manifest_sha256"] = _digest(train_manifest)
    test_manifest["manifest_sha256"] = _digest(test_manifest)
    for output, payload in ((Path(train_output), train_manifest), (Path(test_output), test_manifest)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return train_manifest, test_manifest


def load_video_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported video manifest schema")
    if manifest.get("manifest_sha256") != _digest(manifest):
        raise ValueError("video manifest digest mismatch")
    rows = manifest.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("video manifest must contain non-empty videos")
    ids = [str(row["video_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("video manifest contains duplicate ids")
    numeric_ids: set[int] = set()
    file_names: set[str] = set()
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", ""))
        captions = row.get("captions")
        numeric_id = int(row["numeric_id"])
        file_name = str(row.get("file_name", ""))
        category = int(row["category"])
        if split not in {"train", "dev", "test"}:
            raise ValueError("video manifest contains an invalid split")
        if not isinstance(captions, list) or not captions or any(
            not isinstance(value, str) or not value.strip() for value in captions
        ):
            raise ValueError("video manifest contains invalid captions")
        if numeric_id in numeric_ids or file_name in file_names or not file_name.lower().endswith(".mp4"):
            raise ValueError("video manifest ids and filenames must be unique")
        if not 0 <= category < 20:
            raise ValueError("video manifest category is invalid")
        numeric_ids.add(numeric_id)
        file_names.add(file_name)
        counts[split] = counts.get(split, 0) + 1
    declared = {str(key): int(value) for key, value in manifest.get("selection", {}).get("counts", {}).items()}
    if counts != declared:
        raise ValueError("video manifest split counts mismatch")
    if set(counts) == {"test"} and any(len(row["captions"]) != 1 for row in rows):
        raise ValueError("1K-A test manifest must contain one caption per video")
    return manifest


__all__ = ["SCHEMA", "build_msrvtt_manifests", "load_video_manifest"]
