# run_on_intel.ps1 — the Intel-hardware benchmark bundle, native Windows.
#
#   1. Install Python 3.11 or 3.12 from python.org (tick "Add python.exe to PATH")
#      and Git for Windows. (Python 3.13 is fine too.)
#   2. git clone <repo>   (or copy the folder over)
#   3. In PowerShell:   Set-ExecutionPolicy -Scope Process Bypass
#                       .\mac\intel\run_on_intel.ps1
#      Optional:        .\mac\intel\run_on_intel.ps1 -Device GPU   (or NPU on Core Ultra)
#
# Produces mac\intel\artifacts\e2e_<hostname>.json and whisper_bench_<hostname>.json,
# each stamped with the CPU brand string and the devices OpenVINO sees, and
# appends sections to mac\intel\BENCHMARKS.md. Commit those three files.
param(
  [string]$Device = "",          # "", CPU, GPU, NPU
  [int]$N = 84,
  [switch]$SkipWhisper
)
$ErrorActionPreference = "Stop"
$Mac = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Mac
$env:VOICELINK_PERCEPTION = "0"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
if ($Device) { $env:VOICELINK_OV_DEVICE = $Device }

Write-Host "== 0. environment"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source; if ($py) { $py = "py -3" } }
if (-not $py) { throw "Python not found on PATH. Install from python.org and tick 'Add to PATH'." }
if (-not (Test-Path ".venv")) { & python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt -r requirements-intel.txt pytest
python -c "import json,sys; sys.path.insert(0,'.'); from intel import hardware_fingerprint as h; print(json.dumps(h(), indent=2))"

Write-Host "== 1. artifacts (student IR committed; MiniLM + Whisper IRs regenerated, ~5 min, downloads ~0.5 GB)"
if (-not (Test-Path "intel\artifacts\minilm-ov\openvino_model.bin")) { python intel\convert.py }
if (-not $SkipWhisper -and -not (Test-Path "intel\artifacts\whisper-base.en-ov-int8\openvino_encoder_model.xml")) { python intel\export_whisper.py }

Write-Host "== 2. student head micro-benchmark"
python intel\bench.py --skip-benchmark-app | Out-Null

Write-Host "== 3. end-to-end board latency through the seam"
python intel\bench_e2e.py --n $N
if ($Device -and $Device -ne "CPU") {
  Write-Host "== 3b. head on $Device"
  python intel\bench_e2e.py --n $N --device $Device --backends ov-head,ov-full
}

if (-not $SkipWhisper) {
  Write-Host "== 4. whisper (committed clips, identical audio to the Mac run)"
  try { python intel\bench_whisper.py } catch { Write-Host "whisper bench failed: $_ (not fatal)" }
}

Write-Host "== 5. the seam still serves on this backend"
$env:VOICELINK_STUDENT_BACKEND = "openvino"; $env:VOICELINK_EMBED_BACKEND = "openvino"
python -m pytest -q tests\test_intel.py tests\test_embed_backend.py tests\test_e2e_smoke.py

Write-Host ""
Write-Host "Done. Commit mac\intel\artifacts\e2e_*.json, whisper_bench_*.json and mac\intel\BENCHMARKS.md."
Get-ChildItem intel\artifacts\e2e_*.json, intel\artifacts\whisper_bench_*.json | Select-Object Name, Length
