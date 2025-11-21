from __future__ import annotations

import torch
import torch.nn as nn

class AccessPatternClassifier(nn.Module):
  def __init__(self, input_dim: int, num_classes: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: [batch_size, input_dim] float32 tensor
    returns: [batch_size, num_classes] logits
    """
    return self.net(x)
