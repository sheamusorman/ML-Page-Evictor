from __future__ import annotations

import random
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
  from simulator import PageMeta

EvictionPolicyFn = Callable[
  [List[Optional[int]], Dict[int, int], Dict[int, "PageMeta"], int],
  int,
]

ML_CONTROLLER = None  # set by simulator when using the ML policy
_LAST_PRINTED_POLICY = None

def set_ml_controller(controller) -> None:
  """
  Register the global ML pattern controller used by ml_policy.
  """
  global ML_CONTROLLER
  ML_CONTROLLER = controller


# FIFO

def fifo_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Evict the page that has been in memory the longest.
  Uses meta[page_id].load_time.
  """
  victim_idx: Optional[int] = None
  oldest_load_time: Optional[int] = None

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue
    m = meta[page_id]
    lt = m.load_time
    if oldest_load_time is None or lt < oldest_load_time:
      oldest_load_time = lt
      victim_idx = idx

  if victim_idx is None:
    victim_idx = 0

  return victim_idx


# LRU

def lru_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Evict the least recently used page.
  Uses meta[page_id].last_access_time.
  """
  victim_idx: Optional[int] = None
  oldest_access_time: Optional[int] = None

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue
    m = meta[page_id]
    at = m.last_access_time
    if oldest_access_time is None or at < oldest_access_time:
      oldest_access_time = at
      victim_idx = idx

  if victim_idx is None:
    victim_idx = 0

  return victim_idx


# MRU


def mru_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Evict the most recently used page.
  Opposite of LRU; good for scan-heavy workloads.
  """
  victim_idx: Optional[int] = None
  newest_access_time: Optional[int] = None

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue
    m = meta[page_id]
    at = m.last_access_time
    if newest_access_time is None or at > newest_access_time:
      newest_access_time = at
      victim_idx = idx

  if victim_idx is None:
    victim_idx = 0

  return victim_idx


# Random

def random_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Evict a random resident page.
  """
  indices = [idx for idx, p in enumerate(frames) if p is not None]
  if not indices:
    return 0
  return random.choice(indices)


# LFU

def lfu_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Evict the least frequently used page.
  Tie-breaker: oldest last_access_time among min-count pages.
  """
  victim_idx: Optional[int] = None
  min_count: Optional[int] = None
  oldest_access_time: Optional[int] = None

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue
    m = meta[page_id]
    cnt = m.access_count
    at = m.last_access_time

    if min_count is None or cnt < min_count:
      min_count = cnt
      oldest_access_time = at
      victim_idx = idx
    elif cnt == min_count:
      if oldest_access_time is not None and at < oldest_access_time:
        oldest_access_time = at
        victim_idx = idx

  if victim_idx is None:
    victim_idx = 0

  return victim_idx


# LHD 

def lhd_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Continuous Least Hit Density.
  """
  victim_idx: Optional[int] = None
  min_density: Optional[float] = None
  min_hits: Optional[int] = None
  oldest_access_time: Optional[int] = None

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue

    m = meta[page_id]
    age = time - m.load_time + 1
    if age <= 0:
      age = 1

    hits = m.access_count
    density = hits / age

    if min_density is None or density < min_density:
      min_density = density
      min_hits = hits
      oldest_access_time = m.last_access_time
      victim_idx = idx
    elif density == min_density:
      if min_hits is not None and hits < min_hits:
        min_hits = hits
        oldest_access_time = m.last_access_time
        victim_idx = idx
      elif hits == min_hits:
        at = m.last_access_time
        if oldest_access_time is not None and at < oldest_access_time:
          oldest_access_time = at
          victim_idx = idx

  if victim_idx is None:
    victim_idx = 0

  return victim_idx


# 2Q-style 

def twoq_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Simplified 2Q-like policy.
  """
  once_seen: List[tuple[int, int]] = []
  multi_hit: List[tuple[int, int]] = []

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue
    m = meta[page_id]
    if m.access_count <= 1:
      once_seen.append((idx, m.last_access_time))
    else:
      multi_hit.append((idx, m.last_access_time))

  if once_seen:
    return min(once_seen, key=lambda p: p[1])[0]
  if multi_hit:
    return min(multi_hit, key=lambda p: p[1])[0]
  return 0


# Sequential hybrid (LRU → MRU)

RECENT_WINDOW_FOR_SEQ = 96  # how far back in time a page counts as "recent"

def seq_hybrid_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Hybrid policy for sequential / strided patterns.
  Intuition:
    Identify pages that have been accessed recently (within RECENT_WINDOW_FOR_SEQ).
    Count how many frames currently hold such "recent" pages.
    If recent pages occupy less than half of the frames:
      behave like LRU (evict the least recently used page) so the
      new sequential pages can fill the cache and old pages are flushed.
    Once recent pages occupy at least half of the frames:
      behave like MRU to avoid polluting the cache with further streaming data.
  """
  num_frames = len(frames)

  # Collect "recent" pages based on last_access_time.
  recent_pages = {
    page_id
    for page_id, m in meta.items()
    if time - m.last_access_time <= RECENT_WINDOW_FOR_SEQ
  }

  # Count how many frames currently hold recent pages.
  recent_frame_count = 0
  for page_id in frames:
    if page_id is not None and page_id in recent_pages:
      recent_frame_count += 1

  # If the cache is not yet dominated by recent pages, flush stale ones via LRU.
  if recent_frame_count < num_frames // 2:
    return lru_policy(frames, page_table, meta, time)

  # Once the cache is at least half full of "recent" pages, use MRU semantics
  # to avoid evicting pages that are unlikely to be reused.
  return mru_policy(frames, page_table, meta, time)


# Preserve-cache policy (no eviction)

def preserve_cache_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Policy that indicates the cache contents should be preserved.

  Returning -1 is interpreted by the Simulator as "skip eviction for this
  access and leave the cache unchanged". This is primarily useful when the
  ML controller believes the workload is truly random.
  """
  return -1

# Loop-preserving hybrid policy

LOOP_WINDOW = 128 

def loop_preserve_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Loop-preserving policy.

  Intuition:
    - Treat pages accessed within the last LOOP_WINDOW references as part of
      the current working set (likely the loop body).
    - Evict pages outside this recent set first (stale pages from the
      previous phase).
    - Among candidates, evict the least recently used page.

  This makes it easier for the cache to:
    - flush out pre-loop pages once a loop starts, and
    - keep the loop pages resident if the loop working set fits in cache.
  """
  victim_idx: Optional[int] = None

  # 1) Identify "recent" pages based on last_access_time.
  recent_pages = {
    page_id
    for page_id, m in meta.items()
    if time - m.last_access_time <= LOOP_WINDOW
  }

  # 2) Split frames into "outside recent WS" and "inside recent WS".
  outside_ws: List[tuple[int, int]] = []  # (idx, last_access_time)
  inside_ws: List[tuple[int, int]] = []

  for idx, page_id in enumerate(frames):
    if page_id is None:
      continue
    m = meta[page_id]
    if page_id in recent_pages:
      inside_ws.append((idx, m.last_access_time))
    else:
      outside_ws.append((idx, m.last_access_time))

  # 3) Prefer evicting stale pages from outside the current working set.
  if outside_ws:
    # LRU among outside-WS pages
    victim_idx = min(outside_ws, key=lambda p: p[1])[0]
  elif inside_ws:
    # All frames are part of the current WS; just use LRU over them.
    victim_idx = min(inside_ws, key=lambda p: p[1])[0]
  else:
    # Fallback: no metadata? default to first frame.
    victim_idx = 0

  return victim_idx


# ML policy

def ml_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  ML-driven eviction policy with confidence-gated fallback to LHD.

  LHD is the true working default, only overridden when the model is confident
  in a specific, specialized pattern (A, C, D, or B with high confidence).
  """
  if ML_CONTROLLER is None:
    return lhd_policy(frames, page_table, meta, time)

  bucket_name = ML_CONTROLLER.current_bucket_name()
  probs = None
  if hasattr(ML_CONTROLLER, "current_probs"):
    probs = ML_CONTROLLER.current_probs()

  if bucket_name is None or probs is None:
    return lhd_policy(frames, page_table, meta, time)

  probs = probs.astype("float32")
  max_prob = float(probs.max())
  bucket_idx = int(probs.argmax())

  CONF_MIN_OVERRIDE = 0.60 
  CONF_STRONG_B = 0.70 

  # 1. Start with LHD as the true baseline policy function (Bucket 1, 4, and low-confidence default)
  policy_fn = lhd_policy

  # 2. Check if the model is confident enough AND the pattern requires specialization
  # Note: Bucket 5 (F: Mixed) is never used as an override policy here.
  if max_prob >= CONF_MIN_OVERRIDE and bucket_idx != 5:

    # Bucket 0 (A: Scan/Stride)
    if bucket_idx == 0:
      policy_fn = seq_hybrid_policy

    # Bucket 1 (B: Random) - Only switch to preserve_cache if highly confident
    elif bucket_idx == 1 and max_prob >= CONF_STRONG_B:
      policy_fn = preserve_cache_policy

    # Bucket 2 (C: Hot Set)
    elif bucket_idx == 2:
      policy_fn = twoq_policy

    # Bucket 3 (D: Loop)
    elif bucket_idx == 3:
      policy_fn = loop_preserve_policy

    # Bucket 4 (E: Phase Shift) stays as lhd_policy (default)

  # PRINT WHEN ML POLICY CHANGES 
  global _LAST_PRINTED_POLICY
  effective = policy_fn.__name__
  if _LAST_PRINTED_POLICY != effective:
    print(f"[ML-POLICY] switch → {effective}   at access #{time} (Bucket {bucket_name}, Conf {max_prob:.2f})")
    _LAST_PRINTED_POLICY = effective

  return policy_fn(frames, page_table, meta, time)


_POLICY_REGISTRY: Dict[str, EvictionPolicyFn] = {
  "fifo": fifo_policy,
  "lru": lru_policy,
  "mru": mru_policy,
  "random": random_policy,
  "lfu": lfu_policy,
  "lhd": lhd_policy,
  "2q": twoq_policy,
  "seq_hybrid": seq_hybrid_policy,
  "seq_hybrid_policy": seq_hybrid_policy,  # alias for controller mapping
  "preserve_cache": preserve_cache_policy,
  "loop_preserve": loop_preserve_policy,
  "loop_preserve_policy": loop_preserve_policy,
  "ml": ml_policy,
}


def get_policy(name: str) -> EvictionPolicyFn:
  key = name.lower()
  if key not in _POLICY_REGISTRY:
    raise ValueError(
      f"Unknown policy '{name}'. Available: {list(_POLICY_REGISTRY)}"
    )
  return _POLICY_REGISTRY[key]
