from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from mosaic.video.features import (
    _decode_video_batch,
    _video_path,
    decode_uniform_frames,
    load_video_feature_bundle,
    make_toy_video_feature_bundle,
    uniform_midpoint_indices,
)


class VideoFeatureTests(unittest.TestCase):
    def test_parallel_decode_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(4):
                name = f"video{index}.mp4"
                (root / name).write_bytes(b"fixture")
                rows.append({"file_name": name})

            def decoder(path: Path, *, frames_per_video: int):
                index = int(path.stem.removeprefix("video"))
                time.sleep((3 - index) * 0.005)
                image = np.full((2, 2, 3), index, dtype=np.uint8)
                from PIL import Image

                return [Image.fromarray(image) for _ in range(frames_per_video)], {
                    "repeated_tail_samples": index
                }

            decode_root = Path(str(root).swapcase()) if os.name == "nt" else root
            groups = _decode_video_batch(
                rows,
                decode_root,
                frames_per_video=2,
                decode_workers=4,
                decoder=decoder,
            )
            try:
                self.assertEqual(
                    [int(np.asarray(frames[0])[0, 0, 0]) for frames, _ in groups],
                    [0, 1, 2, 3],
                )
            finally:
                for frames, _ in groups:
                    for image in frames:
                        image.close()
            with self.assertRaises(FileNotFoundError):
                _video_path(root, {"file_name": "../outside.mp4"})

    def test_parallel_decode_closes_successful_siblings_on_failure(self) -> None:
        class TrackedImage:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(3):
                name = f"video{index}.mp4"
                (root / name).write_bytes(b"fixture")
                rows.append({"file_name": name})
            created: list[TrackedImage] = []

            def decoder(path: Path, *, frames_per_video: int):
                index = int(path.stem.removeprefix("video"))
                if index == 1:
                    time.sleep(0.01)
                    raise RuntimeError("decode failed")
                image = TrackedImage()
                created.append(image)
                return [image], {"repeated_tail_samples": 0}

            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                _decode_video_batch(
                    rows,
                    root,
                    frames_per_video=1,
                    decode_workers=3,
                    decoder=decoder,
                )
            self.assertTrue(created)
            self.assertTrue(all(image.closed for image in created))

    def test_uniform_midpoints_are_bounded_and_deterministic(self) -> None:
        np.testing.assert_array_equal(
            uniform_midpoint_indices(12, 4), np.asarray([1, 4, 7, 10])
        )
        np.testing.assert_array_equal(
            uniform_midpoint_indices(2, 4), np.asarray([0, 0, 1, 1])
        )
        with self.assertRaises(ValueError):
            uniform_midpoint_indices(0, 4)

    def test_toy_bundle_is_pickle_free_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_features.npz"
            expected = make_toy_video_feature_bundle(path)
            metadata, arrays = load_video_feature_bundle(path)
            self.assertEqual(metadata, expected)
            self.assertEqual(arrays["frame_features"].shape, (12, 4, 16))
            self.assertEqual(arrays["caption_features"].shape, (24, 16))
            with np.load(path, allow_pickle=False) as bundle:
                self.assertNotEqual(bundle["frame_features"].dtype, np.dtype(object))

    def test_metadata_digest_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_features.npz"
            make_toy_video_feature_bundle(path)
            with np.load(path, allow_pickle=False) as source:
                values = {key: np.asarray(source[key]) for key in source.files}
            metadata = json.loads(str(values["metadata_json"].item()))
            metadata["video_count"] += 1
            values["metadata_json"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "metadata digest mismatch"):
                load_video_feature_bundle(path)

    def test_real_ffmpeg_decoder_returns_bounded_rgb_samples(self) -> None:
        try:
            import imageio_ffmpeg
        except ImportError:  # pragma: no cover - optional environment guard
            self.skipTest("imageio-ffmpeg is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(path), (16, 16), fps=4, codec="libx264", quality=5
            )
            writer.send(None)
            try:
                for value in range(8):
                    frame = np.full((16, 16, 3), value * 24, dtype=np.uint8)
                    writer.send(frame.tobytes())
            finally:
                writer.close()
            images, diagnostics = decode_uniform_frames(path, frames_per_video=4)
            try:
                self.assertEqual(len(images), 4)
                self.assertTrue(all(image.mode == "RGB" and image.size == (16, 16) for image in images))
                self.assertGreater(diagnostics["duration_seconds"], 0)
            finally:
                for image in images:
                    image.close()


if __name__ == "__main__":
    unittest.main()
