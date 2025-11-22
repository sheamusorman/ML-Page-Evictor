from __future__ import annotations

from typing import List
import numpy as np

NUM_FEATURES = 16  # keep this in sync with AccessPatternClassifier input_dim

def compute_window_features(window: List[int]) -> np.ndarray:
  """
  Computes a 16D feature vector for a window of page IDs.

  Features (roughly grouped):

    Locality / uniqueness:
      0: unique_ratio        = unique_pages / window_size
      1: repeat_ratio        = 1 - unique_ratio
      2: entropy             = entropy of page distribution
      3: top3_frac           = fraction of accesses in the top-3 most frequent pages

    Sequential / stride statistics:
      4: sequential_frac     = fraction of diffs == +1
      5: stride_small_frac   = fraction of |diffs| <= 4
      6: mean_abs_diff       = mean |diff|
      7: std_abs_diff        = std |diff|
      8: large_jump_frac     = fraction of |diffs| > 32
      9: backward_frac       = fraction of diffs < 0

    Reuse distance statistics:
      10: reuse_mean         = mean reuse distance (or window_size if none)
      11: reuse_std          = std reuse distance (or 0 if none)
      12: reuse_small_frac   = fraction of reuse distances <= 10
      13: reuse_max          = max reuse distance (or window_size if none)

    Run-length / structure:
      14: max_run_len        = longest run of identical page IDs
      15: run_count_norm     = number of runs / window_size
  """
  n = len(window)
  vals = np.array(window, dtype=np.int64)

  if n == 0:
    return np.zeros(NUM_FEATURES, dtype=np.float32)

  # 1) unique / repeat ratios + entropy + top3 concentration
  unique_vals, counts = np.unique(vals, return_counts=True)
  unique_ratio = len(unique_vals) / float(n)
  repeat_ratio = 1.0 - unique_ratio

  p = counts.astype(np.float64) / float(n)
  entropy = float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0

  sorted_counts = np.sort(counts)[::-1]
  top_k = sorted_counts[:3] if sorted_counts.size >= 3 else sorted_counts
  top3_frac = float(top_k.sum()) / float(n)

  # 2) sequential / stride statistics
  if n > 1:
    diffs = np.diff(vals)
    abs_diffs = np.abs(diffs)

    sequential_frac = float(np.mean(diffs == 1))
    stride_small_frac = float(np.mean(abs_diffs <= 4))

    mean_abs_diff = float(np.mean(abs_diffs))
    std_abs_diff = float(np.std(abs_diffs))

    large_jump_frac = float(np.mean(abs_diffs > 32))
    backward_frac = float(np.mean(diffs < 0))
  else:
    sequential_frac = 0.0
    stride_small_frac = 0.0
    mean_abs_diff = 0.0
    std_abs_diff = 0.0
    large_jump_frac = 0.0
    backward_frac = 0.0

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
    reuse_std = float(np.std(reuse_arr))
    reuse_small_frac = float(np.mean(reuse_arr <= 10))
    reuse_max = float(np.max(reuse_arr))
  else:
    reuse_mean = float(n)
    reuse_std = 0.0
    reuse_small_frac = 0.0
    reuse_max = float(n)

  # 4) run-length / structure
  max_run_len = 1
  run_count = 1  # at least one run if n > 0
  cur_run = 1
  for i in range(1, n):
    if vals[i] == vals[i - 1]:
      cur_run += 1
      if cur_run > max_run_len:
        max_run_len = cur_run
    else:
      run_count += 1
      cur_run = 1

  run_count_norm = float(run_count) / float(n)

  features = np.array([
    unique_ratio,
    repeat_ratio,
    entropy,
    top3_frac,
    sequential_frac,
    stride_small_frac,
    mean_abs_diff,
    std_abs_diff,
    large_jump_frac,
    backward_frac,
    reuse_mean,
    reuse_std,
    reuse_small_frac,
    reuse_max,
    float(max_run_len),
    run_count_norm,
  ], dtype=np.float32)

  return features
