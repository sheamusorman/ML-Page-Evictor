from __future__ import annotations
import os
import random
from typing import Dict, List


def _loop_trace(num_pages: int, length: int) -> List[int]:
  """
  Simple loop over a small working set.
  """
  return [i % num_pages for i in range(length)]


def _locality_shift_trace(
  segment_len: int = 50,
  pages_per_segment: int = 10,
  num_segments: int = 4,
) -> List[int]:
  """
  Working set shifts over time:
  [0..9]*, then [10..19]*, etc.
  """
  trace: List[int] = []
  for s in range(num_segments):
    base = s * pages_per_segment
    for _ in range(segment_len):
      trace.append(base + (len(trace) % pages_per_segment))
  print(f"Generated locality shift trace: {trace}")
  return trace


def _random_mixed_trace(
  num_pages_hot: int = 10,
  num_pages_cold: int = 50,
  length: int = 500,
  hot_prob: float = 0.8,
) -> List[int]:
  """
  Mixed random accesses with a hot subset.
  """
  rng = random.Random()
  hot = list(range(num_pages_hot))
  cold = list(range(num_pages_hot, num_pages_hot + num_pages_cold))

  trace: List[int] = []
  for _ in range(length):
    if rng.random() < hot_prob:
      trace.append(rng.choice(hot))
    else:
      trace.append(rng.choice(cold))

  return trace

def _random_trace(
  num_pages: int = 50,
  length: int = 500,
) -> List[int]:
  """
  Mixed random accesses with a hot subset.
  """
  rng = random.Random()
  pages = list(range(num_pages))

  trace: List[int] = []
  for _ in range(length):
    trace.append(rng.choice(pages))
  return trace

# Predefined traces
_PREDEFINED_TRACES: Dict[str, List[int]] = {
  "loop_small": _loop_trace(num_pages=5, length=200),
  "loop_large": _loop_trace(num_pages=20, length=500),
  "locality_shift": _locality_shift_trace(),
  "mixed_random": _random_mixed_trace(),
  "random": _random_trace(),
}


def get_trace(name: str) -> List[int]:
  """
  Return a trace (list of page IDs).

  If `name` matches a predefined trace key, return that.
  Otherwise, if `name` is a path to a text file, read integers
  from the file (one page ID per line, or space-separated).
  """
  if name in _PREDEFINED_TRACES:
    return list(_PREDEFINED_TRACES[name])

  if os.path.isfile(name):
    pages: List[int] = []
    with open(name, "r") as f:
      for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
          continue
        for token in line.split():
          pages.append(int(token))
    return pages

  raise ValueError(
    f"Unknown trace '{name}'. "
    f"Available predefined traces: {list(_PREDEFINED_TRACES)} "
    f"or provide a valid file path."
  )
