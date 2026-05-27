import torch
import torch.nn as nn


class LearnedGateRefiner(nn.Module):
    """
    Learned refinement gate:
        g = g_hand * sigmoid(MLP(x))

    Input x: [B, 4]
      1) pooled audio feature norm
      2) pooled image feature norm
      3) audio-image cosine similarity
      4) temporal std of per-frame audio norms

    Output:
      factor: [B, 1] in (0, 1)
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Start with relatively high sigmoid output so early training
        # does not over-suppress the hand-crafted gate.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)  # sigmoid(2) ~= 0.88

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))
