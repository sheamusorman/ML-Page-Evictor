from __future__ import annotations

import random
from typing import Dict, List


# ----- Trace generators for different access patterns -----

def _sequential_scan_trace(
  length: int = 2000,
  start_page: int = 0,
) -> List[int]:
  """
  Single forward scan with no repeated pages.
  Represents Bucket A: sequential with essentially no reuse.
  """
  return list(range(start_page, start_page + length))


def _strided_scan_trace(
  length: int = 2000,
  stride: int = 4,
  start_page: int = 0,
) -> List[int]:
  """
  Strided scan with no repeated pages.
  Represents Bucket A: strided access with minimal reuse.
  """
  return [start_page + i * stride for i in range(length)]


def _pure_random_trace(
  num_pages: int = 512,
  length: int = 2000,
  seed: int | None = None,
) -> List[int]:
  """
  Uniform random accesses over a larger page range (no explicit hot set).
  Represents Bucket B: random reads with minimal spatial or temporal locality.
  """
  rng = random.Random(seed)
  trace: List[int] = []
  for _ in range(length):
    trace.append(rng.randrange(num_pages))
  return trace


def _hotset_trace(
  num_pages_hot: int = 16,
  length: int = 2000,
  noise_prob: float = 0.0,
  num_noise_pages: int = 256,
  seed: int | None = None,
) -> List[int]:
  """
  High-locality hot set: repeated accesses within a small page set.

  Represents Bucket C: strong temporal locality confined to a tight hot set.
  Optional noise_prob introduces occasional accesses to a larger cold region.
  """
  rng = random.Random(seed)
  hot = list(range(num_pages_hot))
  cold = list(range(num_pages_hot, num_pages_hot + num_noise_pages))
  trace: List[int] = []

  for _ in range(length):
    if noise_prob > 0.0 and rng.random() < noise_prob:
      trace.append(rng.choice(cold))
    else:
      trace.append(rng.choice(hot))

  return trace


def _loop_trace(
  num_pages: int,
  length: int = 2000,
  stride: int = 1,
  jitter_prob: float = 0.05,
  seed: int | None = None,
) -> List[int]:
  """
  Looping pattern over a working set of num_pages pages.

  Represents Bucket D: cyclic / looping behavior. The loop walks through
  the working set in order, with occasional jitter to avoid being too perfect.
  """
  rng = random.Random(seed)
  pages = [i * stride for i in range(num_pages)]
  trace: List[int] = []
  idx = 0

  for _ in range(length):
    if rng.random() < jitter_prob:
      idx = rng.randrange(num_pages)
    trace.append(pages[idx])
    idx = (idx + 1) % num_pages

  return trace


def _locality_shift_trace(
  segment_len: int = 200,
  pages_per_segment: int = 16,
  num_segments: int = 8,
  gap: int = 16,
  seed: int | None = None,
) -> List[int]:
  """
  Working-set shift pattern.

  Each segment focuses on a contiguous range of pages, then shifts to a
  new range with a configurable gap. Represents Bucket E: phase-shifted
  locality where hot regions move over time.
  """
  rng = random.Random(seed)
  trace: List[int] = []

  for seg in range(num_segments):
    base = seg * (pages_per_segment + gap)
    pages = list(range(base, base + pages_per_segment))
    for _ in range(segment_len):
      trace.append(rng.choice(pages))

  return trace


def _random_mixed_trace(
  num_pages_hot: int = 16,
  num_pages_cold: int = 256,
  length: int = 2000,
  hot_prob: float = 0.8,
  seed: int | None = None,
) -> List[int]:
  """
  Mixed random accesses with a hot subset and a larger cold region.

  Represents another view of Bucket E: hot pages embedded in a larger
  workload with a non-trivial cold background.
  """
  rng = random.Random(seed)
  hot = list(range(num_pages_hot))
  cold = list(range(num_pages_hot, num_pages_hot + num_pages_cold))
  trace: List[int] = []

  for _ in range(length):
    if rng.random() < hot_prob:
      trace.append(rng.choice(hot))
    else:
      trace.append(rng.choice(cold))

  return trace


# ----- Predefined sample traces for simulator / quick inspection -----

_PREDEFINED_TRACES: Dict[str, List[int]] = {
  # Bucket A: strided or sequential with no repeats
  "scan_seq": _sequential_scan_trace(length=512, start_page=0),
  "scan_stride": _strided_scan_trace(length=512, stride=4, start_page=0),

  # Bucket B: random reads (no hot set)
  "random_pure": _pure_random_trace(num_pages=512, length=512, seed=None),

  # Bucket C: strong hot set (high locality only)
  "hotset_small": _hotset_trace(
    num_pages_hot=16,
    length=512,
    noise_prob=0.0,
    #seed=2,
  ),

  # Bucket D: looping / cyclic
  "loop_small": _loop_trace(num_pages=8, length=512, stride=random.Random().randint(1, 4), seed=None),
  "loop_large": _loop_trace(num_pages=32, length=512, stride=random.Random().randint(1, 4), seed=None),

  # Bucket E: phase shifts and hot pages in a larger workload
  "locality_shift": _locality_shift_trace(
    segment_len=64,
    pages_per_segment=16,
    num_segments=4,
    gap=16,
    #seed=5,
  ),
  "mixed_random": _random_mixed_trace(
    num_pages_hot=16,
    num_pages_cold=128,
    length=512,
    hot_prob=0.8,
    #seed=6,
  ),
}


def list_trace_names() -> List[str]:
  """
  Return a list of available predefined trace names.
  """
  return sorted(_PREDEFINED_TRACES.keys())


def get_trace(name: str) -> List[int]:
  """
  Return a predefined trace by name.
  """
  key = name.lower()
  if key not in _PREDEFINED_TRACES:
    raise ValueError(
      f"Unknown trace '{name}'. Available: {list(_PREDEFINED_TRACES.keys())}"
    )
  return _PREDEFINED_TRACES[key]


# ============================================================
# Factory helpers for ML dataset generation
# ============================================================

def make_traces_for_pattern_A(
  num_traces: int,
  length: int = 2000,
) -> List[List[int]]:
  """
  Generate traces for Bucket A: strided or sequential with no repeats.
  """
  traces: List[List[int]] = []
  for i in range(num_traces):
    if i % 2 == 0:
      traces.append(_sequential_scan_trace(length=length, start_page=i * length))
    else:
      traces.append(_strided_scan_trace(length=length, stride=4, start_page=i * length))
  return traces


def make_traces_for_pattern_B(
  num_traces: int,
  length: int = 2000,
) -> List[List[int]]:
  """
  Generate traces for Bucket B: random reads with minimal locality.
  """
  traces: List[List[int]] = []
  for i in range(num_traces):
    traces.append(_pure_random_trace(num_pages=1024, length=length, seed=None))
  return traces


def make_traces_for_pattern_C(
  num_traces: int,
  length: int = 2000,
) -> List[List[int]]:
  """
  Generate traces for Bucket C: high-locality hot set.
  """
  traces: List[List[int]] = []
  for i in range(num_traces):
    traces.append(
      _hotset_trace(
        num_pages_hot=16,
        length=length,
        noise_prob=0.02,
        num_noise_pages=512,
        #seed=200 + i,
      )
    )
  return traces


def make_traces_for_pattern_D(
  num_traces: int,
  length: int = 2000,
) -> List[List[int]]:
  """
  Generate traces for Bucket D: looping / cyclic access.
  """
  traces: List[List[int]] = []
  for i in range(num_traces):
    num_pages = 8 if i % 2 == 0 else 32
    traces.append(
      _loop_trace(
        num_pages=num_pages,
        length=length,
        stride=1,
        jitter_prob=0.05,
        #seed=300 + i,
      )
    )
  return traces


def make_traces_for_pattern_E(
  num_traces: int,
  length: int = 2000,
) -> List[List[int]]:
  """
  Generate traces for Bucket E: hot pages in a larger workload and phase shifts.
  """
  traces: List[List[int]] = []
  half = max(1, num_traces // 2)
  # Phase-shift style traces
  for i in range(half):
    traces.append(
      _locality_shift_trace(
        segment_len=100,
        pages_per_segment=16,
        num_segments=length // (100 * 1),
        gap=16,
        #seed=400 + i,
      )
    )
  # Hot/cold mixed traces
  for i in range(num_traces - half):
    traces.append(
      _random_mixed_trace(
        num_pages_hot=16,
        num_pages_cold=512,
        length=length,
        hot_prob=0.8,
        #seed=450 + i,
      )
    )
  return traces
