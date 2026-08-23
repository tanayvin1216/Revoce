# Intel laptop run — copy-paste steps (Windows, Core Ultra 9 185H)

Goal: produce `e2e_<laptop>.json`, `whisper_bench_<laptop>.json` and an updated
`BENCHMARKS.md`, committed to branch `phase4-distillation`. ~30 minutes, mostly waiting.

## 1. Install (once)
1. Python 3.12: https://www.python.org/downloads/windows/ → "Windows installer (64-bit)".
   In the installer **tick "Add python.exe to PATH"**, then Install.
2. Git: https://git-scm.com/download/win → default options all the way.
3. Check the NPU exists: Start → "Device Manager" → expand **Neural processors** →
   you should see **Intel(R) AI Boost**. If that category is missing, install the
   Intel NPU driver: https://www.intel.com/content/www/us/en/download/794734/ (Meteor Lake).

## 2. Get the code
Open **PowerShell** (Start → type "PowerShell"), then paste:

```powershell
cd $HOME
git clone https://github.com/tanayvin1216/VoiceLink.git
cd VoiceLink
git checkout phase4-distillation
```
Git will open a browser window to sign in to GitHub the first time.

## 3. Run the benchmarks
Paste, one block at a time. The first block downloads ~0.5 GB of models and takes ~10 min.

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\mac\intel\run_on_intel.ps1
```
Look at the first thing it prints: it must say `"cpu": "Intel(R) Core(TM) Ultra 9 185H"`
and `"ov_devices": ["CPU", "GPU", "NPU"]`. If NPU is missing from that list, the driver
from step 1.3 is not installed — keep going anyway, the CPU numbers still count.

Then the NPU and GPU runs (fast, ~2 min each):
```powershell
.\mac\intel\run_on_intel.ps1 -Device NPU -SkipWhisper
.\mac\intel\run_on_intel.ps1 -Device GPU -SkipWhisper
```
If one of these prints a red error, that's fine — copy the error text into a file
`mac\intel\artifacts\npu_error.txt` (or `gpu_error.txt`) and continue.

## 4. Commit and push
```powershell
git add mac/intel/artifacts/*.json mac/intel/artifacts/*.txt mac/intel/BENCHMARKS.md
git commit -m "Intel Core Ultra 9 185H benchmarks: CPU, NPU and GPU runs"
git push
```
(If git asks for your name/email first: `git config --global user.name "Tanay Vinaykya"`
and `git config --global user.email "tanayvinaykya16@gmail.com"`, then repeat the commit.)

Done. Back on the Mac: `git pull`, and Claude Code folds the numbers into the papers.

## If something breaks
- `python` not found → close and reopen PowerShell after installing Python.
- `run_on_intel.ps1 cannot be loaded` → you skipped the `Set-ExecutionPolicy` line.
- pip errors about `torch` → run `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` then rerun the script.
- Anything else → `git add -A mac/intel; git commit -m "partial laptop run"; git push` and send the error text; partial results are still useful.
