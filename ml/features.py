from __future__ import annotations

from typing import List
import numpy as np

def compute_window_features(window: List[int]) -> np.ndarray:
  """
  Computes an 8D feature vector for a window of page IDs.

  Features:
    0: unique_ratio        = unique_pages / window_size
    1: repeat_ratio        = 1 - unique_ratio
    2: sequential_frac     = fraction of diffs == +1
    3: stride_small_frac   = fraction of |diffs| <= 4
    4: reuse_mean          = mean reuse distance (or window_size if none)
    5: reuse_small_frac    = fraction of reuse distances <= 10
    6: entropy             = entropy of page distribution
    7: max_run_len         = longest run of identical page IDs
  """
  n = len(window)
  vals = np.array(window, dtype=np.int64)

  if n == 0:
    return np.zeros(8, dtype=np.float32)

  # 1) unique / repeat ratios
  unique_vals, counts = np.unique(vals, return_counts=True)
  unique_ratio = len(unique_vals) / float(n)
  repeat_ratio = 1.0 - unique_ratio

  # 2) sequential / stride statistics
  if n > 1:
    diffs = np.diff(vals)
    sequential_frac = float(np.mean(diffs == 1))
    stride_small_frac = float(np.mean(np.abs(diffs) <= 4))
  else:
    sequential_frac = 0.0
    stride_small_frac = 0.0

  # 3) approximate reuse distance stats
  last_pos: dict[int, int] = {}
  reuse_dists: list[int] = []
  for i, v in enumerate(vals):
    if v in last_pos:
      reuse_dists.append(i - last_pos[v])
    last_pos[v] = i

  if reuse_dists:
    reuse_arr = np.array(reuse_dists, dtype=np.int64)
    reuse_mean = float(np.mean(reuse_arr))
    reuse_small_frac = float(np.mean(reuse_arr <= 10))
  else:
    reuse_mean = float(n)
    reuse_small_frac = 0.0

  # 4) entropy of page distribution
  p = counts.astype(np.float64) / float(n)
  entropy = float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0

  # 5) max run length (same page)
  max_run_len = 1
  cur_run = 1
  for i in range(1, n):
    if vals[i] == vals[i - 1]:
      cur_run += 1
      if cur_run > max_run_len:
        max_run_len = cur_run
    else:
      cur_run = 1

  features = np.array([
    unique_ratio,
    repeat_ratio,
    sequential_frac,
    stride_small_frac,
    reuse_mean,
    reuse_small_frac,
    entropy,
    float(max_run_len),
  ], dtype=np.float32)

  return features
