from __future__ import annotations

from typing import List, Tuple
import numpy as np

import traces
from ml.features import compute_window_features

# ----- Access pattern labels (buckets) -----

# A: Strided or sequential with no repeats
# B: Random reads
# C: High-locality hot set
# D: Looping (cyclic)
# E: Hot pages in larger workload + phase shift
# F: Mixed / ambiguous pattern windows
ACCESS_PATTERN_LABELS = {
  "A": 0,
  "B": 1,
  "C": 2,
  "D": 3,
  "E": 4,
  "F": 5,
}

WINDOW_SIZE = 48
STRIDE = 24

TRACE_LENGTH = 2000
NUM_TRACES_PER_PATTERN = 20

MIXED_SEGMENT_LEN = 24
MIXED_TOTAL_LEN = 2000
NUM_MIXED_TRACES = 20


# ----- Window helpers -----

def make_windows(
  trace: List[int],
  window_size: int,
  stride: int,
) -> List[List[int]]:
  """
  Slice a trace into overlapping windows.
  """
  windows: List[List[int]] = []
  n = len(trace)
  i = 0
  while i + window_size <= n:
    windows.append(trace[i : i + window_size])
    i += stride
  return windows


def windows_for_traces(
  traces_list: List[List[int]],
  pattern_label: int,
  window_size: int = WINDOW_SIZE,
  stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Build (X, y) for a list of traces and a single access pattern label.
  """
  X_list: list[np.ndarray] = []
  y_list: list[int] = []

  for t in traces_list:
    win_list = make_windows(t, window_size, stride)
    for w in win_list:
      X_list.append(compute_window_features(w))
      y_list.append(pattern_label)

  if not X_list:
    return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.int64)

  X = np.stack(X_list, axis=0)
  y = np.array(y_list, dtype=np.int64)
  return X, y


# ----- Base dataset for A–E (clean pattern windows) -----

def build_base_dataset() -> Tuple[np.ndarray, np.ndarray]:
  """
  Build a base dataset from traces that cleanly represent buckets A–E.
  """
  # Generate traces for each pattern using the factory helpers in traces.py.
  traces_A = traces.make_traces_for_pattern_A(
    num_traces=NUM_TRACES_PER_PATTERN,
    length=TRACE_LENGTH,
  )
  traces_B = traces.make_traces_for_pattern_B(
    num_traces=NUM_TRACES_PER_PATTERN,
    length=TRACE_LENGTH,
  )
  traces_C = traces.make_traces_for_pattern_C(
    num_traces=NUM_TRACES_PER_PATTERN,
    length=TRACE_LENGTH,
  )
  traces_D = traces.make_traces_for_pattern_D(
    num_traces=NUM_TRACES_PER_PATTERN,
    length=TRACE_LENGTH,
  )
  traces_E = traces.make_traces_for_pattern_E(
    num_traces=NUM_TRACES_PER_PATTERN,
    length=TRACE_LENGTH,
  )

  X_A, y_A = windows_for_traces(traces_A, ACCESS_PATTERN_LABELS["A"])
  X_B, y_B = windows_for_traces(traces_B, ACCESS_PATTERN_LABELS["B"])
  X_C, y_C = windows_for_traces(traces_C, ACCESS_PATTERN_LABELS["C"])
  X_D, y_D = windows_for_traces(traces_D, ACCESS_PATTERN_LABELS["D"])
  X_E, y_E = windows_for_traces(traces_E, ACCESS_PATTERN_LABELS["E"])

  X_list = [X_A, X_B, X_C, X_D, X_E]
  y_list = [y_A, y_B, y_C, y_D, y_E]

  X_nonempty = [X for X in X_list if X.shape[0] > 0]
  y_nonempty = [y for y in y_list if y.shape[0] > 0]

  if not X_nonempty:
    raise RuntimeError("No base windows generated for access patterns A–E.")

  X_base = np.concatenate(X_nonempty, axis=0)
  y_base = np.concatenate(y_nonempty, axis=0)
  return X_base, y_base


# ----- Mixed dataset for F (ambiguous / combination windows) -----

def _build_mixed_trace(
  trace_a: List[int],
  trace_b: List[int],
  segment_len: int = MIXED_SEGMENT_LEN,
  total_len: int = MIXED_TOTAL_LEN,
) -> List[int]:
  """
  Build a mixed trace by alternating short segments from trace_a and trace_b.

  This produces windows that contain a combination of two distinct access
  patterns, representing Bucket F (mixed / ambiguous behavior).
  """
  out: List[int] = []
  idx_a = 0
  idx_b = 0
  len_a = len(trace_a)
  len_b = len(trace_b)

  while len(out) < total_len and len_a > 0 and len_b > 0:
    # Segment from A
    seg_a = trace_a[idx_a : idx_a + segment_len]
    if len(seg_a) < segment_len:
      idx_a = 0
      seg_a = trace_a[idx_a : idx_a + segment_len]
    out.extend(seg_a)
    idx_a = (idx_a + segment_len) % len_a

    if len(out) >= total_len:
      break

    # Segment from B
    seg_b = trace_b[idx_b : idx_b + segment_len]
    if len(seg_b) < segment_len:
      idx_b = 0
      seg_b = trace_b[idx_b : idx_b + segment_len]
    out.extend(seg_b)
    idx_b = (idx_b + segment_len) % len_b

  if len(out) > total_len:
    out = out[:total_len]

  return out


def build_mixed_dataset_for_F() -> Tuple[np.ndarray, np.ndarray]:
  """
  Build windows representing Bucket F (mixed / ambiguous patterns).

  This is done by:
    - generating base traces for A–E
    - constructing new traces that alternate segments from two different buckets
    - labeling all resulting windows as F
  """
  pattern_F = ACCESS_PATTERN_LABELS["F"]

  # Generate a few traces per pattern specifically for mixing.
  traces_A = traces.make_traces_for_pattern_A(num_traces=NUM_MIXED_TRACES, length=TRACE_LENGTH)
  traces_B = traces.make_traces_for_pattern_B(num_traces=NUM_MIXED_TRACES, length=TRACE_LENGTH)
  traces_C = traces.make_traces_for_pattern_C(num_traces=NUM_MIXED_TRACES, length=TRACE_LENGTH)
  traces_D = traces.make_traces_for_pattern_D(num_traces=NUM_MIXED_TRACES, length=TRACE_LENGTH)
  traces_E = traces.make_traces_for_pattern_E(num_traces=NUM_MIXED_TRACES, length=TRACE_LENGTH)

  pattern_groups = {
    "A": traces_A,
    "B": traces_B,
    "C": traces_C,
    "D": traces_D,
    "E": traces_E,
  }

  # Pairs of buckets to mix for F
  mix_pairs: list[tuple[str, str]] = [
    ("A", "C"),
    ("B", "E"),
    ("D", "E"),
  ]

  X_list: list[np.ndarray] = []
  y_list: list[int] = []

  for key_a, key_b in mix_pairs:
    group_a = pattern_groups[key_a]
    group_b = pattern_groups[key_b]
    # Use up to NUM_MIXED_TRACES mixed traces per pair.
    for i in range(NUM_MIXED_TRACES):
      trace_a = group_a[i % len(group_a)]
      trace_b = group_b[i % len(group_b)]
      mixed = _build_mixed_trace(
        trace_a,
        trace_b,
        segment_len=MIXED_SEGMENT_LEN,
        total_len=MIXED_TOTAL_LEN,
      )
      windows = make_windows(mixed, WINDOW_SIZE, STRIDE)
      for w in windows:
        X_list.append(compute_window_features(w))
        y_list.append(pattern_F)

  if not X_list:
    return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.int64)

  X_F = np.stack(X_list, axis=0)
  y_F = np.array(y_list, dtype=np.int64)

  rng = np.random.default_rng(0)
  n_F = X_F.shape[0]
  keep = max(n_F // 3, 1)
  idx = rng.choice(n_F, size=keep, replace=False)
  X_F = X_F[idx]
  y_F = y_F[idx]

  return X_F, y_F



# ----- Full dataset builder -----

def build_full_dataset() -> Tuple[np.ndarray, np.ndarray]:
  """
  Build the full dataset for access pattern classification (A–F).
  """
  X_base, y_base = build_base_dataset()
  X_F, y_F = build_mixed_dataset_for_F()

  if X_F.shape[0] == 0:
    X_full = X_base
    y_full = y_base
  else:
    X_full = np.concatenate([X_base, X_F], axis=0)
    y_full = np.concatenate([y_base, y_F], axis=0)

  return X_full, y_full


if __name__ == "__main__":
  X, y = build_full_dataset()
  print("X shape:", X.shape)
  print("y shape:", y.shape)
  np.save("X_access_pattern.npy", X)
  np.save("y_access_pattern.npy", y)
  print("Saved X_access_pattern.npy and y_access_pattern.npy")
