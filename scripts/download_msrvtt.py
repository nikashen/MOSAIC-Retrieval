from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path


REPO = "friedrichor/MSR-VTT"
COMMIT = "c1af215a96934854f42683c19c51391aaee6f962"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/{COMMIT}"
FILES = {
    "MSRVTT_Videos.zip": {
        "url": f"{BASE}/MSRVTT_Videos.zip",
        "bytes": 2_188_992_999,
        "sha256": "be4935000f7f9470ff9852d833b2ec808fcfaeb5beb3b2c08d25894b59352196",
    },
    "msrvtt_train_9k.json": {
        "url": f"{BASE}/msrvtt_train_9k.json",
        "bytes": 14_192_792,
        "sha256": "97ff86055d8fcec43ca2ce948f89a02fa529f3b26fc16845f623e1dbaed5db44",
    },
    "msrvtt_test_1k.json": {
        "url": f"{BASE}/msrvtt_test_1k.json",
        "bytes": 341_953,
        "sha256": "6e937737375f77bffc939ba336001e0328fafdc1c844868a5e158ce4de4ce81f",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, insecure: bool = False) -> None:
    args = [
        "curl.exe" if os.name == "nt" else "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
    ]
    if insecure:
        args.insert(1, "-k")
    if destination.exists():
        args += ["-C", "-"]
    args += ["-o", str(destination), url]
    subprocess.run(args, check=True)


def safe_extract(archive_path: Path, output_root: Path) -> Path:
    output_root = output_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_root / member.filename).resolve()
            if os.path.commonpath((str(target), str(output_root))) != str(output_root):
                raise ValueError(f"unsafe ZIP member: {member.filename}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        archive.extractall(output_root)
    candidates = [path for path in output_root.iterdir() if path.is_dir() and list(path.glob("*.mp4"))]
    if len(candidates) != 1:
        raise ValueError(f"expected one extracted video directory, found {candidates}")
    return candidates[0]


def resolve_video_root(root: Path, value: str | os.PathLike[str]) -> Path:
    """Resolve new relative markers and safely migrate historical absolute ones."""

    root = root.resolve()
    recorded = Path(value)
    candidates = [recorded.resolve()] if recorded.is_absolute() else [(root / recorded).resolve()]
    portable = (root / "video").resolve()
    if portable not in candidates:
        candidates.append(portable)
    for candidate in candidates:
        if candidate.is_dir() and os.path.commonpath((str(candidate), str(root))) == str(root):
            return candidate
    raise FileNotFoundError(f"MSR-VTT extracted video directory is unavailable: {candidates}")


def verify_extracted_crc(archive_path: Path, video_root: Path) -> dict[str, object]:
    """Match every extracted MP4 to the pinned ZIP member size and CRC32."""

    video_root = Path(video_root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (member for member in archive.infolist() if not member.is_dir() and member.filename.lower().endswith(".mp4")),
            key=lambda member: Path(member.filename).name,
        )
    if len(members) != 10_000:
        raise ValueError(f"expected 10,000 MP4 members, found {len(members)}")
    names: list[str] = []
    manifest = hashlib.sha256()
    total_bytes = 0
    for member in members:
        name = Path(member.filename).name
        path = (video_root / name).resolve()
        if os.path.commonpath((str(path), str(video_root))) != str(video_root) or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != member.file_size:
            raise ValueError(f"extracted size mismatch for {name}")
        crc = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                crc = zlib.crc32(chunk, crc)
        crc &= 0xFFFFFFFF
        if crc != member.CRC:
            raise ValueError(f"extracted CRC mismatch for {name}")
        names.append(name)
        total_bytes += member.file_size
        manifest.update(f"{name}\t{member.file_size}\t{member.CRC:08x}\n".encode())
    return {
        "video_root": video_root.name,
        "videos": len(members),
        "total_bytes": total_bytes,
        "video_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "member_manifest_sha256": manifest.hexdigest(),
        "crc32_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned MSR-VTT 1K-A mirror")
    parser.add_argument("--root", type=Path, default=Path("data/raw/msrvtt"))
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="explicitly disable TLS certificate verification (not recommended)",
    )
    args = parser.parse_args()
    raise SystemExit(
        "Automatic MSR-VTT download is disabled in the public snapshot because the "
        "historical mirror declares no dataset license. Provide a locally authorized "
        "copy and follow docs/MSRVTT_PROTOCOL.md."
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = [name for name in FILES if not (args.skip_videos and name.endswith(".zip"))]
    checksums: dict[str, object] = {}
    for name in selected:
        spec = FILES[name]
        path = root / name
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            download_file(str(spec["url"]), path, insecure=args.insecure)
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != spec["bytes"] or actual_sha != spec["sha256"]:
            raise RuntimeError(
                f"{name} verification failed: size={actual_size}, sha256={actual_sha}"
            )
        checksums[name] = {
            "url": spec["url"],
            "bytes": actual_size,
            "sha256": actual_sha,
        }
    if not args.skip_videos:
        archive = root / "MSRVTT_Videos.zip"
        marker = root / ".mosaic_msrvtt_extracted.json"
        if args.force_extract and marker.is_file():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            target = resolve_video_root(root, previous["video_root"])
            if target.parent == root and target.is_dir():
                shutil.rmtree(target)
            marker.unlink()
        if marker.is_file():
            extract_info = json.loads(marker.read_text(encoding="utf-8"))
            if extract_info.get("archive_sha256") != FILES["MSRVTT_Videos.zip"]["sha256"]:
                raise ValueError("extraction marker archive hash mismatch")
            video_root = resolve_video_root(root, extract_info["video_root"])
        else:
            video_root = safe_extract(archive, root)
        extracted = verify_extracted_crc(archive, video_root)
        extracted["archive_sha256"] = FILES["MSRVTT_Videos.zip"]["sha256"]
        marker.write_text(
            json.dumps(extracted, indent=2)
            + "\n",
            encoding="utf-8",
        )
        checksums["extracted"] = extracted
    report = {
        "schema_version": "mosaic.msrvtt_download.v1",
        "source": f"{REPO}@{COMMIT}",
        "tls_verification_disabled": bool(args.insecure),
        "license_boundary": (
            "The pinned mirror does not declare a dataset license. Use only for local "
            "academic/non-commercial evaluation; do not redistribute videos."
        ),
        "files": checksums,
    }
    (root / "download_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
