"""The student model — a small, ONNX-clean MLP ranker.

Input : a single MiniLM context embedding (context_dim,) for "<scene> || <transcript>".
Output: one logit per fixed-vocabulary item (vocab_size,). Sigmoid + BCE makes this
        a multi-label "which of these phrases belong on the board right now" head.

Inference board = the top-k highest-scoring vocab items, mapped back to words. The
diversity the Phase-3 ruler measures therefore comes from what the student learned
to rank highly — NOT from any post-hoc MMR — so a mode-collapsed student is caught.

Only Linear / ReLU / Dropout are used: every op is in the ONNX opset and the
OpenVINO operator set, which is why the export-first checkpoint can pass before a
single training step.
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
        """Top-k vocab indices for ONE context embedding (1D or (1,D))."""
        if context_emb.dim() == 1:
            context_emb = context_emb.unsqueeze(0)
        logits = self.forward(context_emb)[0]
        k = min(k, logits.shape[0])
        return torch.topk(logits, k).indices.tolist()
