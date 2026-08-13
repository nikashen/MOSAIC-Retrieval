from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from mosaic.data import manifest_sha256


def selection_key(image_id: int, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}:{int(image_id)}".encode()).digest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(image_id: int, destination: Path, retries: int = 5) -> tuple[int, str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://images.cocodataset.org/train2017/{image_id:012d}.jpg"
    context = ssl._create_unverified_context()
    for attempt in range(retries):
        temporary = destination.with_suffix(".jpg.part")
        try:
            with urllib.request.urlopen(url, timeout=60, context=context) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return image_id, sha256_file(destination), destination.stat().st_size
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 >= retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an untouched deterministic COCO external final")
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/annotations/captions_train2017.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/external_final_train2017"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/coco_external_final_1k.json"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--salt", default="mosaic-external-final-v1-frozen")
    args = parser.parse_args()
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in payload["images"]}
    selected = sorted(image_rows, key=lambda value: selection_key(value, args.salt))[: args.count]
    if len(selected) != args.count:
        raise ValueError("annotation catalog is smaller than requested final set")
    captions: dict[int, list[str]] = defaultdict(list)
    selected_set = set(selected)
    for row in payload["annotations"]:
        image_id = int(row["image_id"])
        if image_id in selected_set:
            text = " ".join(str(row["caption"]).split())
            if text:
                captions[image_id].append(text)
    missing_captions = [value for value in selected if not captions[value]]
    if missing_captions:
        raise ValueError(f"selected images without captions: {missing_captions[:5]}")
    hashes: dict[int, tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for image_id in selected:
            destination = args.image_root / f"{image_id:012d}.jpg"
            if destination.is_file():
                with Image.open(destination) as image:
                    image.verify()
                hashes[image_id] = (sha256_file(destination), destination.stat().st_size)
            else:
                futures[pool.submit(download_one, image_id, destination)] = image_id
        for number, future in enumerate(as_completed(futures), start=1):
            image_id, digest, size = future.result()
            hashes[image_id] = (digest, size)
            if number % 100 == 0:
                print(f"downloaded {number}/{len(futures)}")
    records = [
        {
            "image_id": int(image_id),
            "file_name": f"{image_id:012d}.jpg",
            "captions": captions[image_id],
            "split": "external_final",
            "image_sha256": hashes[image_id][0],
            "image_bytes": hashes[image_id][1],
        }
        for image_id in sorted(selected)
    ]
    manifest: dict[str, Any] = {
        "schema_version": "mosaic.manifest.v1",
        "dataset": "COCO-2017-train-external-final-1k",
        "annotation_file": str(args.annotations.resolve()),
        "image_root": str(args.image_root.resolve()),
        "split": {
            "method": "lowest sha256(salt:image_id) over official train2017 image catalog",
            "salt": args.salt,
            "selection_count": args.count,
            "source_catalog_images": len(image_rows),
            "counts": {"external_final": len(records)},
        },
        "images": records,
        "evidence_boundary": {
            "adapter_training_images": False,
            "dev_selection_images": False,
            "internal_diagnostic_test_images": False,
            "pretrained_clip_may_have_seen_public_coco": True,
        },
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "images": len(records), "captions": sum(len(row["captions"]) for row in records), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

