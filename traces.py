from __future__ import annotations
import os
import random
from typing import Dict, List

TRACE_LENGTH = 1000
MAX_PAGE_ID = 500
CACHE_SIZE = 30 # Typical number of frames in memory

def _loop_trace(num_pages: int, length: int) -> List[int]:
  """
  Simple loop over a small working set.
  """
  start = random.randint(0, TRACE_LENGTH - 50)
  return [i % num_pages for i in range(length)]

def _locality_shift_trace(
  segment_len: int = TRACE_LENGTH,
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
  return trace

def _shifting_workingset(
  segment_len: int = TRACE_LENGTH,
  pages_per_working_set: int = 50,
  num_accesses: int = 100,
) -> List[int]:
  """
  Working set shifts over time with some randomness.
  """
  trace: List[int] = []
  for _ in range(TRACE_LENGTH // num_accesses):
    base = random.randint(0, MAX_PAGE_ID - pages_per_working_set)
    for _ in range(num_accesses):
      trace.append(base + random.randint(0, pages_per_working_set))
  print(f"Generated shifting working set trace: {trace}")
  return trace

def _random_mixed_trace(
  num_pages_hot: int = 20,
  num_pages_cold: int = 80,
  length: int = TRACE_LENGTH,
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
  num_pages: int = 100,
  length: int = TRACE_LENGTH,
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
  "loop_small": _loop_trace(num_pages=5, length=TRACE_LENGTH),
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
