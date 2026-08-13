from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_public_release", ROOT / "scripts" / "verify_public_release.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load public release verifier")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicReleaseTests(unittest.TestCase):
    def test_public_snapshot_passes(self) -> None:
        payload = MODULE.verify_public_release()
        self.assertEqual("pass", payload["status"])
        self.assertEqual(0, payload["raw_images"])
        self.assertEqual(0, payload["captions_or_queries"])


if __name__ == "__main__":
    unittest.main()
