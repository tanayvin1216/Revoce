"""Phase-4 distillation v1.

Teacher  = the diverse pool + MMR selection in rank_vocabulary (slow LLM path).
Student  = a small MLP over a MiniLM context embedding that scores a FIXED
           vocabulary directly (no LLM at inference), built only from standard
           layers that export cleanly to ONNX / OpenVINO.

The seam (rank_vocabulary) is never imported as a dependency by the student at
inference time — the whole point is that the student REPLACES the LLM pool-gen.
"""
