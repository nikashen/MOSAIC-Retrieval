$ErrorActionPreference = "Stop"
$python = if ($env:MOSAIC_PYTHON) { $env:MOSAIC_PYTHON } else { "python" }
& $python -B -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -B scripts\verify_public_release.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -B docs\verify_pages.py
exit $LASTEXITCODE
