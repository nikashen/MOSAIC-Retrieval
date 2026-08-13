"""MOSAIC-Retrieval audited multimodal experiment package."""

import os

# Must be set before the first CUDA context is created; seed ordering alone is
# insufficient for deterministic cuBLAS matrix multiplications on CUDA >=10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.0.0"
