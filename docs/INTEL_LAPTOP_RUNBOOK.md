# VoiceLink — Intel Core Ultra benchmark runbook

**Run this on the Windows laptop (Core Ultra 9 185H). Paste one block at a time,
check the gate under it, stop if a gate fails.**

Total time: ~20 min, most of it the one-time downloads in Block 2.

Everything you need is in this file. Open PowerShell — not cmd, not WSL.

---

## Block 1 — set the repo path (edit this line, run once)

Change the path to wherever the repo is on this laptop, then paste the whole block.

```powershell
$Repo = "C:\Users\<you>\VoiceLink"
Set-Location $Repo
Set-ExecutionPolicy -Scope Process Bypass -Force
git fetch origin
git reset --hard origin/main
git log --oneline -1
```

**Gate:** the last line must print `466b954` or newer. If it prints `3db7a81`,
the fetch didn't work — the history was rewritten, so try
`git fetch origin; git reset --hard origin/main` again.

> If you don't have the repo on this laptop yet, use this instead of the above:
> ```powershell
> $Repo = "C:\Users\<you>\VoiceLink"
> git clone https://github.com/tanayvin1216/VoiceLink.git $Repo
> Set-Location $Repo
> Set-ExecutionPolicy -Scope Process Bypass -Force
> ```

---

## Block 2 — the main run (~15 min, downloads ~0.5 GB once)

```powershell
Set-Location $Repo
.\mac\intel\run_on_intel.ps1
```

This creates the venv, installs everything, builds the MiniLM IR, **proves the
OpenVINO embedder actually loads**, and runs the head micro-benchmark, the
end-to-end board benchmark and Whisper.

**Gate — it must NOT throw.** Two throws are possible and both are informative:

- `MiniLM IR was not built` → the export failed. Scroll up for the
  `minilm export FAILED (rc=...)` block and **send me that text**.
- `embed backend resolved to 'torch', not openvino` → the IR built but won't
  load. Scroll up for the `[embed] OpenVINO MiniLM unavailable (...)` line and
  **send me that line**.

**Also check in the output:**
- `ov-full` shows `embed=openvino` (this is the whole point — last run it said
  `torch` and the number was meaningless)
- no line saying `[bench_e2e] held-out rows unavailable`
- no `error` rows in the backend table

---

## Block 3 — GPU and NPU

```powershell
Set-Location $Repo\mac
.\.venv\Scripts\Activate.ps1
python intel\bench_e2e.py --n 84 --no-md --device GPU --backends torch,ov-head,ov-full
python intel\bench_e2e.py --n 84 --no-md --device NPU --backends torch,ov-head,ov-full
```

**Gate:** two files appear — `e2e_<host>_gpu.json` and `e2e_<host>_npu.json`.
An `error` row for NPU is a legitimate result, not a failure; keep it.

**Expect GPU and NPU to look almost identical to CPU.** `OpenVinoEmbedder` pins
MiniLM to the CPU, so `--device` only moves the ~0.1 ms student head out of an
~18 ms board. That is the honest finding and we will report it as such.

---

## Block 4 — the FP32 vs INT8 ablation

Measures whether the OpenVINO student's different board choices come from
quantization or from something else.

```powershell
Set-Location $Repo\mac
Rename-Item intel\artifacts\student_int8.xml student_int8.xml.off
python intel\bench_e2e.py --n 84 --no-md --backends torch,ov-head,ov-full
$src = Get-ChildItem intel\artifacts\e2e_*.json |
       Where-Object { $_.Name -notmatch '_(gpu|npu|fp32)\.json$' } |
       Sort-Object LastWriteTime | Select-Object -Last 1
Copy-Item $src.FullName intel\artifacts\e2e_fp32.json
Rename-Item intel\artifacts\student_int8.xml.off student_int8.xml
```

**Gate:** `student_int8.xml` must exist again at the end. If the middle command
failed, run the rename-back on its own:

```powershell
Rename-Item $Repo\mac\intel\artifacts\student_int8.xml.off student_int8.xml
```

---

## Block 5 — check nothing got clobbered, then commit

```powershell
Set-Location $Repo
git status
```

**Gate — read this carefully. `git status` must show ONLY new/modified
`e2e_*.json`, `whisper_bench_*.json` and `head_bench_*.json` files.**

If any of these appear as modified, the run damaged them — undo before committing:

```powershell
git checkout -- mac/intel/BENCHMARKS.md mac/evaluation/phase4/student.onnx mac/evaluation/phase4/student.onnx.data mac/evaluation/phase4/student.pt mac/evaluation/phase4/vocab.json
```

Then commit and push:

```powershell
Set-Location $Repo
git add mac/intel/artifacts/e2e_*.json mac/intel/artifacts/whisper_bench_*.json mac/intel/artifacts/head_bench_*.json
git status
git commit -m "core ultra: openvino embeddings, gpu and npu rows, and the fp32 ablation"
git push origin main
```

---

## Block 6 — teacher on this machine (OPTIONAL, ~20 min, downloads ~6 GB once)

Only worth doing if we want a speedup ratio measured entirely on Intel. Without
it the ratio stays explicitly Mac-only, which is also fine — we just have to say
so. **Skip this if you are short on time.**

This is safe to run on its own: `teacher_slice.py` does NOT touch
`teacher_data.jsonl` and does NOT retrain anything. Do not reach for `-Train`
to get it.

```powershell
Set-Location $Repo\mac
.\.venv\Scripts\Activate.ps1
python intel\teacher_slice.py --n 10
```

**Gate:** writes `intel\artifacts\teacher_slice_<host>.jsonl`. Include it in
the `git add` in Block 5 if you run it.

---

## Never do these

- **`-Train`** — retrains the student on this laptop. It is what put a 59-class
  `student.onnx` into the repo and made a whole benchmark run measure the wrong
  model. The script now labels it DANGER.
- **Committing `BENCHMARKS.md` from this laptop** — both machines' numbers get
  merged on the Mac. The laptop's copy will be wrong.
- **Committing anything named `student_*`** — those artifacts are committed and
  correct; a local rebuild is always a regression.

---

## What to send back if something fails

Copy the terminal output from the failing block. Specifically:

| If this failed | Send |
|---|---|
| Block 2, MiniLM | the whole `minilm export FAILED (rc=...)` block |
| Block 2, embed probe | the `[embed] OpenVINO MiniLM unavailable (...)` line |
| Block 3, NPU | the `error` row text — it's a result, but I need the wording |
| Anything else | the last ~30 lines |

If everything passed, just push and say so — I'll pull the JSON and merge both
machines into `BENCHMARKS.md` and `VERIFIED_NUMBERS.md` on the Mac.

---

## Why this run matters

The previous Core Ultra run had two silent failures: it measured torch
embeddings while reporting an OpenVINO configuration, and it built its FP32
numbers against a 59-class model instead of the real 95-class one. Both are
fixed in the code now — the script hard-fails instead of falling back quietly.

The number this run is really after is the `ov-full` board latency with
embeddings genuinely on OpenVINO. On the Mac that step took the board from
9.7 ms to 5.5 ms — a 1.76× win, and by far the largest OpenVINO effect anywhere
in the project. The Core Ultra has never actually measured it.
