"""The v3 student — a bi-encoder scorer with an OPEN output space.

Why this replaces the 94-way classifier
---------------------------------------
The v2 student was `Linear(769 -> V)`: one output column per vocabulary item.
That forces a closed vocabulary, and a closed vocabulary has to be earned by
frequency — an item needed `min_freq=3` teacher boards to get a column at all.
Measured on the v2 teacher corpus, that floor discarded ~95% of the distinct
phrases the teacher ever produced. Anything situation-specific ("sugar", a
person's name, "my aunt") is below the floor by construction, so it can never
appear on a board no matter how relevant it is. That showed up in the numbers
as dynamic_overlap 0.056: the student kept the teacher's generic function
skeleton and almost none of its concrete word choice.

This model scores a (context, phrase) PAIR instead of owning a column per
phrase:

    score(c, p) = <f_ctx(c), f_phrase(p)>

Two towers project a 769-d context and a 384-d phrase embedding into a shared
latent space; the score is their dot product. Nothing in the architecture
depends on the size or the membership of the candidate set, so:

  * every phrase the teacher ever emitted is representable (no frequency floor);
  * candidates extracted live from the transcript are scorable;
  * names from the on-device personal-context store are scorable;
  * the candidate bank can grow later WITHOUT retraining or re-exporting.

Inference cost stays low because the phrase tower is run once, offline, over
the whole bank and cached. Serving a board is one context-tower forward pass
(769 -> 256 -> 128) plus a single matmul against the cached bank.

Export discipline is unchanged: LayerNorm / Linear / ReLU / Dropout / MatMul are
all opset-17 and all in the OpenVINO operator set. The two towers export
separately, which is also what lets the phrase tower be quantized once and the
context tower be the only thing on the serving hot path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Tower(nn.Module):
    """LayerNorm -> Linear -> ReLU -> Dropout -> Linear. Export-clean."""

    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiEncoderRanker(nn.Module):
    """Scores (context, phrase) pairs in a shared latent space.

    context_dim: 769  (scene 384 + transcript 384 + is_question 1)
    phrase_dim:  384  (MiniLM sentence embedding of the phrase)
    """

    def __init__(self, context_dim: int = 769, phrase_dim: int = 384,
                 hidden_dim: int = 256, proj_dim: int = 128,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.phrase_dim = phrase_dim
        self.proj_dim = proj_dim
        self.context_tower = _Tower(context_dim, hidden_dim, proj_dim, dropout)
        self.phrase_tower = _Tower(phrase_dim, hidden_dim, proj_dim, dropout)

    # -- the two halves, exported separately -------------------------------

    def encode_context(self, context_emb: torch.Tensor) -> torch.Tensor:
        """(B, context_dim) -> (B, proj_dim). The serving hot path."""
        return self.context_tower(context_emb)

    def encode_phrase(self, phrase_emb: torch.Tensor) -> torch.Tensor:
        """(N, phrase_dim) -> (N, proj_dim). Run offline over the bank."""
        return self.phrase_tower(phrase_emb)

    # -- scoring ------------------------------------------------------------

    def forward(self, context_emb: torch.Tensor,
                phrase_emb: torch.Tensor) -> torch.Tensor:
        """(B, context_dim) x (N, phrase_dim) -> (B, N) scores."""
        return self.encode_context(context_emb) @ self.encode_phrase(phrase_emb).T

    def score_against_bank(self, context_emb: torch.Tensor,
                           bank_proj: torch.Tensor) -> torch.Tensor:
        """(B, context_dim) x PRE-PROJECTED (N, proj_dim) -> (B, N).

        This is what inference actually calls: the phrase tower never runs at
        serve time, only this matmul against the cached bank.
        """
        return self.encode_context(context_emb) @ bank_proj.T

    @torch.no_grad()
    def top_indices(self, context_emb: torch.Tensor, bank_proj: torch.Tensor,
                    k: int) -> list[int]:
        """Top-k bank indices for ONE context embedding (1D or (1, D))."""
        if context_emb.dim() == 1:
            context_emb = context_emb.unsqueeze(0)
        scores = self.score_against_bank(context_emb, bank_proj)[0]
        k = min(k, scores.shape[0])
        return torch.topk(scores, k).indices.tolist()


class ContextTowerOnly(nn.Module):
    """Export wrapper: the context tower as a standalone single-input graph.

    ONNX/OpenVINO conversion and NNCF calibration both want one clean input.
    The phrase tower gets the same treatment via PhraseTowerOnly.
    """

    def __init__(self, model: BiEncoderRanker) -> None:
        super().__init__()
        self.tower = model.context_tower

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        return self.tower(context_emb)


class PhraseTowerOnly(nn.Module):
    def __init__(self, model: BiEncoderRanker) -> None:
        super().__init__()
        self.tower = model.phrase_tower

    def forward(self, phrase_emb: torch.Tensor) -> torch.Tensor:
        return self.tower(phrase_emb)
