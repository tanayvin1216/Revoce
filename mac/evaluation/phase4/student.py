"""The student model — a small, ONNX-clean MLP ranker.

Input : a single MiniLM context embedding (context_dim,) for "<scene> || <transcript>".
Output: one logit per DYNAMIC-vocabulary item (vocab_size = V_dyn). Sigmoid + BCE makes
        this a multi-label "which of these phrases belong on the board right now" head.

v2: LayerNorm is prepended to normalise the MiniLM embeddings before the projection.
    LayerNorm is in ONNX opset 17 and the OpenVINO operator set, so the export-first
    checkpoint still passes. The core safety words are assembled by the caller (via
    assemble_board) — this model scores only the dynamic vocabulary.

Inference board = assemble_board(GUARANTEED_CORE, top-k_dyn from this model, k).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StudentRanker(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, vocab_size: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        """(B, context_dim) -> (B, vocab_size) logits."""
        return self.net(context_emb)

    @torch.no_grad()
    def board_indices(self, context_emb: torch.Tensor, k: int) -> list[int]:
        """Top-k dynamic vocab indices for ONE context embedding (1D or (1,D))."""
        if context_emb.dim() == 1:
            context_emb = context_emb.unsqueeze(0)
        logits = self.forward(context_emb)[0]
        k = min(k, logits.shape[0])
        return torch.topk(logits, k).indices.tolist()
