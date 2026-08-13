from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


FILES = {
    "val2017.zip": {
        "url": "https://images.cocodataset.org/zips/val2017.zip",
        # The current CDN object is a valid ZIP but its byte-level digest differs
        # from the historical COCO page's multipart ETag. Freeze both the
        # observed MD5 and SHA-256 so a partial/proxy-corrupted download fails.
        "md5": "442b8da7639aecaf257c1dceb8ba8c80",
        "sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
        "extract_dir": "val2017",
    },
    "annotations_trainval2017.zip": {
        "url": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "md5": "f4bbac642086de4f52a3fdda2de5fa2c",
        "sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
        "extract_dir": "annotations",
    },
}


def md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((str(target), str(destination))) != str(destination):
                raise ValueError(f"zip path traversal: {member.filename}")
        archive.extractall(destination)


def download(url: str, destination: Path, *, insecure: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "curl.exe",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "0",
    ]
    if insecure:
        args.insert(1, "-k")
    if destination.exists():
        args += ["-C", "-"]
    args += ["-o", str(destination), url]
    print("$", " ".join(args))
    subprocess.run(args, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify official COCO val/captions files")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-annotations", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="explicitly disable TLS certificate verification (not recommended)",
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = []
    if not args.skip_images:
        selected.append("val2017.zip")
    if not args.skip_annotations:
        selected.append("annotations_trainval2017.zip")
    if not selected:
        raise SystemExit("nothing selected")
    checksum_path = root.parent / "checksums.json"
    checksums: dict[str, dict[str, str]] = (
        json.loads(checksum_path.read_text(encoding="utf-8"))
        if checksum_path.is_file()
        else {}
    )
    for name in selected:
        spec = FILES[name]
        archive = root / name
        download(spec["url"], archive, insecure=args.insecure)
        actual = md5(archive)
        actual_sha256 = sha256(archive)
        print(f"{name}: md5={actual}")
        if actual != spec["md5"]:
            raise RuntimeError(f"MD5 mismatch for {name}: expected {spec['md5']}, got {actual}")
        if spec.get("sha256") and actual_sha256 != spec["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {name}: expected {spec['sha256']}, got {actual_sha256}")
        destination = root / spec["extract_dir"]
        marker = destination / ".mosaic_extracted"
        if args.force_extract and destination.exists():
            shutil.rmtree(destination)
        if not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, root)
            marker.write_text(f"{name}\n{actual}\n", encoding="utf-8")
        checksums[name] = {"url": spec["url"], "md5": actual, "sha256": actual_sha256, "bytes": str(archive.stat().st_size)}
    checksum_path.write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checksums, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
