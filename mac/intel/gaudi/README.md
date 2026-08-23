# Intel Gaudi — what runs here and what does not

Gaudi (2/3) is a PyTorch training/serving accelerator. **OpenVINO does not target
it**, so the deployment path (`intel/convert.py`, `quantize.py`, `bench*.py`,
`run_on_intel.sh`) belongs on a Xeon 6 / Core Ultra box, not here.

What Gaudi is for in VoiceLink (rubric: "AI-specific hardware" and "Intel across
the lifecycle"):

| Lifecycle stage | Script | Device flag |
|---|---|---|
| Data — teacher board generation (Qwen2.5-3B-Instruct via transformers) | `evaluation/phase4/gen_teacher.py` | `VOICELINK_RANKER=hf VOICELINK_DEVICE=hpu` |
| Modelling — student distillation | `evaluation/phase4/train.py` | `VOICELINK_DEVICE=hpu` (checkpoint is saved on CPU) |
| Evaluation — gates | `evaluation/phase4/evaluate.py --teacher-samples 0` | CPU |

`run_on_gaudi.sh` does all of the above inside Intel's own container
(`vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0`,
the image the Intel cloud quick start lists). Pass `--inside --gen` to also
generate new teacher rows.

## Getting a Gaudi machine (status 2026-08-23)

Every Intel document (Habana quick start, `developer.habana.ai/intel-ai-cloud`,
the Gaudi *get access* page) points at `console.cloud.intel.com`, and that host
**does not resolve** (NXDOMAIN from public resolvers) as of today. The portal has
moved or been renamed; find the current entry point in a browser from
`intel.com/content/www/us/en/developer/platform/gaudi/get-access.html` (Intel
403s non-browser clients). Other documented routes: AWS EC2 DL1 (Gaudi 1), IBM
Cloud (Gaudi 3). Account creation, SSH key upload and credits/coupon entry are
human steps; once `ssh` works, the run is one command.

## Honesty

Nothing in this directory has been executed on Gaudi hardware yet. The HF
ranker and the device-aware trainer were verified on CPU/MPS on the Mac; `hpu`
selection imports `habana_frameworks.torch.core` and is untested until a
machine exists. Do not cite Gaudi in any document until `run_on_gaudi.sh` has
produced a log.
