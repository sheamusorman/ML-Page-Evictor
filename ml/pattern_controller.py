from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np
import torch

from ml.features import compute_window_features
from ml.access_pattern_model import AccessPatternClassifier


class PatternController:
  """
  Online controller that:
    - observes page IDs,
    - classifies the current access pattern (buckets A–F),
    - selects an underlying eviction policy name.

  This class is stateless w.r.t. the cache; it only looks at access history.
  """

  def __init__(
    self,
    model_path: str = "access_pattern_classifier.pt",
    window_size: int = 192,
    history_len: int = 64,
    device: Optional[torch.device] = None,
  ) -> None:
    self.window_size = window_size
    self.history_len = history_len

    self.buffer: Deque[int] = deque(maxlen=window_size)
    self.recent_preds: Deque[int] = deque(maxlen=history_len)

    self.device = device or torch.device("cpu")

    self.model = AccessPatternClassifier(input_dim=16, num_classes=6)
    state = torch.load(model_path, map_location=self.device)
    self.model.load_state_dict(state)
    self.model.to(self.device)
    self.model.eval()

    # Bucket indices → policy names
    # 0: A (scan / stride, no repeats)     → seq_hybrid_policy
    # 1: B (random)                        → preserve_cache
    # 2: C (tight hot set)                 → 2Q
    # 3: D (loop)                          → loop_preserve
    # 4: E (phase shift / hot+background)  → LHD
    # 5: F (mixed / transition)            → keep last stable policy
    self.bucket_to_policy = {
      0: "seq_hybrid_policy",
      1: "lhd",
      2: "2q",
      3: "loop_preserve",
      4: "lhd",
      5: None,
    }

    self.bucket_names = ["A", "B", "C", "D", "E", "F"]
    self.current_policy: str = "lhd"  # default/fallback policy
    self.current_bucket_idx: Optional[int] = None

    self.last_probs: Optional[torch.Tensor] = None

  # public API 

  def observe(self, page_id: int) -> None:
    """
    Feed a new page reference into the controller.

    Once the buffer is full, this recomputes the access-pattern prediction
    on the most recent window and updates the smoothed bucket estimate.
    """
    self.buffer.append(page_id)
    if len(self.buffer) < self.window_size:
      return

    window = list(self.buffer)
    feats = compute_window_features(window)
    x = torch.from_numpy(feats.reshape(1, -1)).to(self.device)

    with torch.no_grad():
      logits = self.model(x)
      probs = torch.softmax(logits, dim=1)[0]

    # Store raw probs for confidence gating in ml_policy.
    self.last_probs = probs.detach().cpu()

    pred_idx = self._pick_bucket_from_probs(probs)
    self.recent_preds.append(pred_idx)
    self.current_bucket_idx = self._smoothed_bucket()

  def current_bucket(self) -> Optional[int]:
    """
    Return the current bucket index (0–5) or None if not enough history.
    """
    return self.current_bucket_idx

  def current_bucket_name(self) -> Optional[str]:
    """
    Return the current bucket name ('A'..'F') or None.
    """
    if self.current_bucket_idx is None:
      return None
    return self.bucket_names[self.current_bucket_idx]

  def current_policy_name(self) -> str:
    """
    Return the eviction policy name that should be active right now.

    Bucket F (index 5) is treated as a transition / ambiguous region.
    In that case, keep using the last stable policy.
    """
    idx = self.current_bucket_idx
    if idx is None:
      return self.current_policy

    mapped = self.bucket_to_policy.get(idx)
    if mapped is None:
      # Bucket F or anything unassigned: stay with last stable choice.
      return self.current_policy

    self.current_policy = mapped
    return self.current_policy

  def current_probs(self) -> Optional[np.ndarray]:
    """
    Return the last raw probability vector as a numpy array, or None.
    """
    if self.last_probs is None:
      return None
    return self.last_probs.numpy()

  # internals 

  def _pick_bucket_from_probs(self, probs: torch.Tensor) -> int:
    """
    Convert raw class probabilities into a bucket index, with special handling
    for B and F so they are only chosen when the model is clearly confident.

    probs: tensor of shape [6] for buckets A..F.
    """
    # Default argmax
    pred = int(torch.argmax(probs).item())

    # Indices: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F
    # Target: We want B (random) and F (mixed) to only be chosen if dominant.
    # Rationale: When the pattern is ambiguous (B or F), we prefer to default
    # to an exploitable pattern (A, C, D, E) or rely on LHD's safety net.
    if pred in (1, 5):  # B or F
      pred_prob = float(probs[pred].item())

      if pred_prob < 0.7:
        allowed = [0, 2, 3, 4]
        sub = probs[allowed]
        alt = int(torch.argmax(sub).item())
        return allowed[alt]

    return pred

  def _smoothed_bucket(self) -> Optional[int]:
    """
    Majority vote over the last few raw predictions to avoid flapping.
    """
    if not self.recent_preds:
      return None

    arr = np.array(list(self.recent_preds), dtype=np.int64)
    counts = np.bincount(arr, minlength=6)
    return int(np.argmax(counts))
