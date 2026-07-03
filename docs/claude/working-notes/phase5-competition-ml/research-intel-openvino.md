# Research Brief: Intel/OpenVINO Technical Path for VoiceLink

## Question
What is the concrete, honest engineering path to bring real Intel technology (OpenVINO, NNCF, optimum-intel, oneAPI, DevCloud/Tiber) into VoiceLink's pipeline — given the current stack is faster-whisper + mlx-vlm Qwen2.5-VL-3B + an ONNX-exported MLP student ranker, all running on Apple Silicon behind the single `rank_vocabulary` seam?

## Key Findings

**1. OpenVINO on Apple Silicon**
- `pip install openvino` ships native arm64 wheels (macosx_11_0_arm64) as of the 2026.2.1 release; Python 3.10–3.14 supported (confidence: HIGH — [PyPI](https://pypi.org/project/openvino/)).
- Apple Silicon support is via the **ARM CPU plugin only** — no GPU/NPU plugin for Mac (Apple's GPU is not an OpenVINO target; NPU plugin is Intel-only silicon). This has been true since the arm-plugin work landed (confidence: HIGH — [GitHub discussion #13308](https://github.com/openvinotoolkit/openvino/discussions/13308), [PyPI arm64 wheel issue #11554](https://github.com/openvinotoolkit/openvino/issues/11554)). I could not pull the exact current system-requirements matrix text for macOS arm64 plugin support (WebFetch on `docs.openvino.ai/systemrequirements` returned only navigation, not content) — **treat as MEDIUM/unverified for the precise 2026.2 doc wording**, but corroborated by multiple community sources.
- **Practical implication**: CPU-plugin inference on Apple Silicon is a legitimate way to *run and validate* OpenVINO IR models during development (honest claim: "runs identically on ARM via OpenVINO's CPU plugin"), but it is NOT the Intel hardware acceleration path. Judges will want to see numbers on real Intel silicon (Core Ultra/Xeon) — see DevCloud section.

**2. Whisper via OpenVINO**
- Export command (confidence: HIGH, verified against optimum-intel docs and search corroboration — [optimum-intel export docs](https://huggingface.co/docs/optimum-intel/en/openvino/export)):
  ```
  optimum-cli export openvino --trust-remote-code --model openai/whisper-base.en \
      --task automatic-speech-recognition-with-past whisper-base.en-ov
  ```
  INT8 weight compression: add `--weight-format int8`. Full int8 weight+activation quantization (needs calibration dataset) uses `--quant-mode int8 --dataset librispeech --num-samples 32 --smooth-quant-alpha 0.9` (this exact pattern is documented for `whisper-large-v3-turbo` in optimum-intel docs, so it generalizes to whisper-base.en — MEDIUM confidence on base.en specifically, HIGH on the mechanism).
  Alternative: `openvino_genai.WhisperPipeline` for a native C++/Python inference API post-export.
- Published NNCF blog benchmarks (2.1x–6.1x speedup, 1–2% WER degradation) are for **large-v2/distil-large-v2 only**; no base.en-specific published numbers found (confidence: MEDIUM on the mechanism transferring, LOW/unverified on exact numbers for base.en — flag explicitly to judges as "we benchmarked it ourselves" rather than citing a paper number). Generic OpenVINO-vs-PyTorch claims of "1.4x–5.1x faster" appear in OpenVINO's own blog/notebooks (confidence: MEDIUM — [OpenVINO blog](https://blog.openvino.ai/blog-posts/optimizing-whisper-and-distil-whisper-for-speech-recognition-with-openvino-and-nncf)).
- faster-whisper (current backend, CTranslate2) and OpenVINO WhisperPipeline are different runtimes — swapping is a backend change behind the seam, not a rewrite of transcription logic.

**3. Qwen2.5-VL via OpenVINO GenAI**
- Qwen2.5-VL (including the 3B variant) is supported by `openvino_genai.VLMPipeline` as of OpenVINO 2026.2 (confidence: MEDIUM-HIGH). Export:
  ```
  optimum-cli export openvino -m Qwen/Qwen2.5-VL-3B-Instruct --weight-format int4 Qwen2.5-VL-3B-Instruct-ov-int4
  ```
  Usage: `openvino_genai.VLMPipeline("Qwen2.5-VL-3B-Instruct-ov-int4", "CPU")` (or "GPU"/"NPU" on Intel hardware) — [Medium: OpenVINO 2026.2 multimodal](https://medium.com/openvino-toolkit/running-qwen3-6-35b-a3b-on-an-ai-pc-openvino-2026-2-brings-a-new-local-multimodal-llm-experience-e3a09aa26103), [HF llmware/Qwen2.5-VL-3B-Instruct-ov-int4-npu](https://huggingface.co/llmware/Qwen2.5-VL-3B-Instruct-ov-int4-npu) shows a pre-quantized 3B checkpoint already exists.
- **Known issue, now resolved**: GitHub issue [openvinotoolkit/openvino.genai#1845](https://github.com/openvinotoolkit/openvino.genai/issues/1845) — "Unsupported qwen2_5_vl VLM model type" — was closed via PR #2140. Older openvino.genai builds will fail on qwen2_5_vl; **the fix requires a sufficiently recent openvino.genai version** (confidence: HIGH that the issue existed and was fixed; MEDIUM/unverified on the exact minimum version — check `pip show openvino-genai` at implementation time).
- Note: the current runtime checkpoint is `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (MLX 4-bit quant, Apple-specific format — see `mac/perception/config.py:34`). This is a **different quantization format** from the OpenVINO INT4 export; you cannot reuse the MLX weights directly — re-export from the original `Qwen/Qwen2.5-VL-3B-Instruct` HF checkpoint via optimum-cli.
- Fallback if the pinned openvino.genai predates the fix: LLaVA-OV or SmolVLM2 are commonly cited OpenVINO-GenAI-first alternatives (LOW confidence, contingency only).

**4. Student ranker (ONNX → OpenVINO IR) + NNCF PTQ**
- `export_onnx.py` already produces a clean opset-17 ONNX with dynamic batch axis and no custom ops — exactly the "export-first checkpoint" pattern that de-risks OpenVINO conversion.
- Conversion command (confidence: HIGH — [OpenVINO model-prep docs](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html)):
  ```python
  import openvino as ov
  ov_model = ov.convert_model("student.onnx")
  ov.save_model(ov_model, "student.xml")
  ```
  or CLI: `ovc student.onnx --output_model student_ov/`
- NNCF PTQ for the MLP (confidence HIGH — [NNCF PTQ usage](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md)):
  ```python
  import nncf
  quantized_model = nncf.quantize(ov_model, nncf.Dataset(calibration_embeddings))
  ```
  Calibration set = a few hundred real 769-d embeddings from eval scenarios (`v2_scenarios.jsonl`) — cheap, gives an honest INT8-vs-FP32 accuracy table on the actual eval metric.
- MiniLM (sentence-transformers) quantization: same `nncf.quantize` PTQ flow after `optimum-cli export openvino --model sentence-transformers/all-MiniLM-L6-v2 --library sentence_transformers` (documented flag — confidence HIGH).
- `benchmark_app -m student.xml -d CPU -hint latency` / `-hint tput` gives honest min/avg/max latency + throughput for FP32 and INT8 IRs — the standard, judge-legible before/after table (confidence HIGH — [benchmark_app docs](https://docs.openvino.ai/nightly/get-started/learn-openvino/openvino-samples/benchmark-tool.html)).

**5. NNCF specifics**
- Two distinct APIs, don't conflate: `nncf.quantize()` = full PTQ (weights + activations, needs calibration data) for small/medium models (the MLP, MiniLM, Whisper encoder/decoder). `nncf.compress_weights()` = weight-only compression (INT8/INT4 mixed) for large models (Qwen2.5-VL LLM backbone) where activation quantization isn't worth the accuracy cost (confidence HIGH — [NNCF weight compression docs](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md)).
- NNCF's PTQ API works directly on ONNX, OpenVINO IR, PyTorch, and TensorFlow objects; OpenVINO-native is the "preferred backend" per NNCF's docs.
- Accuracy-drop measurement: run the existing eval harness once on FP32 IR and once on INT8 IR, report the delta as the honest number — do not cite Intel's Whisper-large blog numbers for base.en/MLP models.

**6. Intel DevCloud / Tiber AI Cloud (2026)**
- The old unconditional free "Standard" compute tier tokens are gone (2026 community complaints — [Intel Community thread](https://community.intel.com/t5/Intel-Tiber-Developer-Cloud/What-happened-to-the-free-tier/td-p/1614678)).
- What remains free: **JupyterHub / "Training and Workshops" batch-mode notebook sessions** on 4th Gen Xeon + Data Center GPU Max, and a "Standard – Free" registration tier per search summaries — but multiple 2026 threads describe friction (no credits, can't launch instances) for students without a university partnership (confidence: MEDIUM — conflicting signals, flagged as a contradiction).
- Sign-up: register at `cloud.intel.com` / `ai.cloud.intel.com`, SSH keys, pay or redeem coupon, launch from Hardware Catalog (Xeon 6 128-core and Core Ultra AI PC dev-kits listed in Preview catalog) — [Intel Tiber article](https://www.intel.com/content/www/us/en/developer/articles/news/tiber-ai-cloud-expanded-ai-compute.html). Direct load of `ai.cloud.intel.com` failed (TLS error) — **unverified; needs a manual browser check**.
- **Unverified**: whether a hackathon/competition credit program is active mid-2026. Recommend emailing Intel developer relations or checking the competition's sponsor materials.

**7. Honest end-to-end Intel story for judges**
- Dev/iteration on the Mac using OpenVINO's ARM CPU plugin — proves functional correctness of the OpenVINO-exported models (Whisper IR, Qwen2.5-VL IR, student-ranker IR) behind the `rank_vocabulary` seam, zero code changes to swap backends (what the seam was built for).
- NNCF quantization (PTQ for student+MiniLM+Whisper, weight-compression for Qwen2.5-VL) benchmarked with `benchmark_app`, producing our own before/after latency + accuracy tables — no borrowed Intel numbers.
- Final validation on **actual Intel hardware** (Core Ultra or Xeon via Tiber AI Cloud, if access secured) is the strongest single addition for "Intel tech across lifecycle" — flagged as the biggest open risk since free-tier access is inconsistent in 2026.
- Frame explicitly: "OpenVINO/optimum-intel/NNCF are the actual inference and compression stack; Mac ARM CPU plugin was the dev loop; Tiber/DevCloud (or a borrowed Intel laptop) produced the final numbers."

## Recommendations
1. Start with the student ranker (smallest, lowest-risk artifact — already ONNX opset 17). Convert to IR, `nncf.quantize` PTQ with real eval-set embeddings, benchmark FP32 vs INT8 via `benchmark_app` on the Mac's ARM CPU plugin first.
2. Export MiniLM to OpenVINO IR via `optimum-cli --library sentence_transformers`, quantize, benchmark — closes the full embedding→ranking chain in OpenVINO.
3. Export whisper-base.en via `optimum-cli export openvino --task automatic-speech-recognition-with-past`, apply `--weight-format int8`, benchmark against the faster-whisper baseline on the same audio clips as the eval harness.
4. Attempt Qwen2.5-VL-3B export from the original HF checkpoint (not MLX 4-bit) with `--weight-format int4`; pin a recent `openvino-genai`; verify VLMPipeline loads before investing in benchmarking.
5. Attempt Tiber AI Cloud signup early (highest-risk external dependency, longest lead time) — do not block the pipeline on it.
6. Wire OpenVINO backends into `rank_vocabulary` behind the same seam as an alternate config flag (e.g. `VOICELINK_BACKEND=openvino`), never a parallel code path.

## Risks
- Tiber/DevCloud free access may not materialize before the deadline — treat Mac ARM CPU plugin numbers as the primary honest deliverable, Intel-hardware numbers as a stretch goal labeled "pending cloud access".
- Qwen2.5-VL OpenVINO GenAI support is recent (2026.2) and had an open bug — verify installed `openvino-genai` version before committing engineering time.
- Published NNCF speedup/accuracy numbers are for large-v2 Whisper, not base.en — run the actual eval harness instead of citing them.
- MLX 4-bit Qwen2.5-VL checkpoint is not portable to OpenVINO INT4 — re-export from source HF weights required; two separately-quantized model files to maintain.

## Unknowns
- Exact OpenVINO 2026.2 system-requirements text for macOS arm64 plugin support.
- Whether Intel Tiber AI Cloud currently offers student/competition credits (conflicting reports; direct site fetch failed with TLS error — manual browser check needed).
- Minimum `openvino-genai` version containing the qwen2_5_vl fix (issue #1845).
- Whether a competition-specific Intel sponsorship/credit program exists.

## Sources
- [openvino PyPI package](https://pypi.org/project/openvino/)
- [OpenVINO archive install docs, macOS](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-macos.html)
- [OpenVINO system requirements](https://docs.openvino.ai/systemrequirements)
- [GitHub discussion: Apple M1/M2 arm plugin](https://github.com/openvinotoolkit/openvino/discussions/13308)
- [GitHub issue: Apple Silicon PyPI wheels](https://github.com/openvinotoolkit/openvino/issues/11554)
- [optimum-intel export documentation](https://huggingface.co/docs/optimum-intel/en/openvino/export)
- [optimum-intel GitHub repo](https://github.com/huggingface/optimum-intel)
- [Medium: OpenVINO 2026.2 multimodal LLM experience](https://medium.com/openvino-toolkit/running-qwen3-6-35b-a3b-on-an-ai-pc-openvino-2026-2-brings-a-new-local-multimodal-llm-experience-e3a09aa26103)
- [HF: Qwen2.5-VL-3B-Instruct-ov-int4-npu](https://huggingface.co/llmware/Qwen2.5-VL-3B-Instruct-ov-int4-npu)
- [GitHub issue #1845: Unsupported qwen2_5_vl](https://github.com/openvinotoolkit/openvino.genai/issues/1845)
- [NNCF GitHub repo](https://github.com/openvinotoolkit/nncf)
- [NNCF post-training quantization usage docs](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md)
- [NNCF weight compression usage docs](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
- [OpenVINO model conversion to IR docs](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html)
- [OpenVINO benchmark_app tool docs](https://docs.openvino.ai/nightly/get-started/learn-openvino/openvino-samples/benchmark-tool.html)
- [OpenVINO blog: Optimizing Whisper/Distil-Whisper with NNCF](https://blog.openvino.ai/blog-posts/optimizing-whisper-and-distil-whisper-for-speech-recognition-with-openvino-and-nncf)
- [Intel Tiber AI Cloud](https://ai.cloud.intel.com/) (fetch failed — TLS error, unverified)
- [Intel community: free tier discussion](https://community.intel.com/t5/Intel-Tiber-Developer-Cloud/What-happened-to-the-free-tier/td-p/1614678)
- [Intel: Tiber AI Cloud expanded compute](https://www.intel.com/content/www/us/en/developer/articles/news/tiber-ai-cloud-expanded-ai-compute.html)

Local files reviewed: `CLAUDE.md`, `mac/evaluation/phase4/export_onnx.py`, `mac/perception/config.py`.
