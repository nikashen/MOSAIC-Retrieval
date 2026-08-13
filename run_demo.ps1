$ErrorActionPreference = "Stop"
$python = if ($env:MOSAIC_PYTHON) { $env:MOSAIC_PYTHON } else { "python" }
& $python -m http.server 8764 --directory docs
