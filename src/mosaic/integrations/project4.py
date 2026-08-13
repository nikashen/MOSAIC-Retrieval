from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mosaic.features import load_feature_bundle


SCHEMA = "mosaic.project4_content_vector.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_id_mapping(path: Path) -> dict[int, int]:
    """Load an explicit COCO-content-id -> project-4-video-id mapping."""

    path = Path(path)
    mapping: dict[int, int] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = ((int(key), int(value)) for key, value in payload.items())
        elif isinstance(payload, list):
            rows = ((int(row["content_id"]), int(row["video_id"])) for row in payload)
        else:
            raise ValueError("mapping JSON must be an object or list of objects")
        for source, target in rows:
            if source in mapping and mapping[source] != target:
                raise ValueError(f"duplicate source id {source}")
            mapping[source] = target
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"content_id", "video_id"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("mapping CSV requires content_id,video_id columns")
            for row in reader:
                source, target = int(row["content_id"]), int(row["video_id"])
                if source in mapping and mapping[source] != target:
                    raise ValueError(f"duplicate source id {source}")
                mapping[source] = target
    if not mapping or len(mapping) != len(set(mapping.values())):
        raise ValueError("mapping must be non-empty and one-to-one")
    return mapping


def export_content_vectors(
    feature_bundle: Path,
    output: Path,
    *,
    mapping: Path | None = None,
    allow_identity_demo: bool = False,
    encoder_version: str = "mosaic-coco5k-v1",
) -> dict[str, Any]:
    feature_bundle = Path(feature_bundle)
    with np.load(feature_bundle, allow_pickle=False) as probe:
        keys = set(probe.files)
    if keys == {"content_id", "content_vector", "metadata_json"}:
        with np.load(feature_bundle, allow_pickle=False) as bundle:
            source_ids = np.asarray(bundle["content_id"], dtype=np.int64)
            vectors = np.asarray(bundle["content_vector"], dtype=np.float32)
            metadata = json.loads(str(bundle["metadata_json"].item()))
        modality_value = 3
        source_scope = str(metadata.get("vector_scope", "trained_multimodal_item_vector"))
        metadata_sha = _sha256_file(feature_bundle)
    else:
        metadata, arrays = load_feature_bundle(feature_bundle)
        source_ids = arrays["image_ids"].astype(np.int64, copy=False)
        vectors = np.asarray(arrays["image_features"], dtype=np.float32)
        modality_value = 1
        source_scope = "zero_shot_image_only_feature_bundle"
        metadata_sha = metadata.get("metadata_sha256")
    if mapping is None:
        if not allow_identity_demo:
            raise ValueError(
                "an explicit content_id->video_id mapping is required; "
                "use --allow-identity-demo only for an isolated contract smoke test"
            )
        target_ids = source_ids.copy()
        scope = "contract_smoke_identity_ids_not_project4_catalog"
        mapping_sha = None
    else:
        id_map = load_id_mapping(mapping)
        missing = [int(value) for value in source_ids if int(value) not in id_map]
        if missing:
            raise ValueError(f"mapping is missing {len(missing)} content ids")
        target_ids = np.asarray([id_map[int(value)] for value in source_ids], dtype=np.int64)
        scope = "explicit_project4_mapping"
        mapping_sha = _sha256_file(mapping)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    masks = np.full((vectors.shape[0],), modality_value, dtype=np.uint8)
    out_metadata = {
        "schema_version": SCHEMA,
        "encoder_version": encoder_version,
        "source_feature_metadata_sha256": metadata_sha,
        "source_vector_scope": source_scope,
        "source_ids_sha256": hashlib.sha256(source_ids.tobytes()).hexdigest(),
        "mapping_sha256": mapping_sha,
        "scope": scope,
        "target_contract": {
            "id_field": "video_id",
            "vector_field": "dense_content_features",
            "dtype": "float32",
            "l2_normalized": True,
            "modality_mask": "1=image, 2=text, 3=both",
        },
        "safe_npz": "allow_pickle_false",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        video_id=target_ids,
        content_vector=vectors,
        modality_mask=masks,
        metadata_json=np.asarray(json.dumps(out_metadata, ensure_ascii=False)),
    )
    return out_metadata


__all__ = ["SCHEMA", "export_content_vectors", "load_id_mapping"]

