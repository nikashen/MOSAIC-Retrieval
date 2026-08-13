from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from mosaic.data import build_toy_manifest
from mosaic.features import make_toy_feature_bundle


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    image_root = root / "data" / "raw" / "toy"
    image_root.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        image = Image.new("RGB", (64, 64), (index * 19 % 255, 80, 140))
        draw = ImageDraw.Draw(image)
        draw.text((5, 25), str(index), fill=(255, 255, 255))
        image.save(image_root / f"toy_{index:03d}.png")
    build_toy_manifest(root / "data" / "processed" / "toy_manifest.json", image_root=image_root)
    make_toy_feature_bundle(root / "artifacts" / "toy_features.npz")
    print("toy manifest and features ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

