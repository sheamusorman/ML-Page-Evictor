from __future__ import annotations

import random
from typing import Dict, List, Optional, Callable

from simulator import PageMeta

EvictionPolicyFn = Callable[
  [List[Optional[int]], Dict[int, int], Dict[int, "PageMeta"], int],
  int,
]

ML_CONTROLLER = None  # set by simulator when using the ML policy


def set_ml_controller(controller) -> None:
  """
  Register the global ML pattern controller used by ml_policy.
  """
  global ML_CONTROLLER
  ML_CONTROLLER = controller


# ----- FIFO -----

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


# ----- LRU -----

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


# ----- MRU -----

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


# ----- Random -----

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


# ----- LFU -----

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


# ----- LHD -----


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


# ----- 2Q-style -----

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


# ----- CLOCK ----- Note, this requires sim.clock_hand to be set up still

def clock_policy(sim, page_id):
  """
  CLOCK (Second-Chance) page replacement.
  Uses sim.clock_hand and a per-frame reference bit.
  """
  if page_id in sim.page_table:
    frame = sim.page_table[page_id]
    sim.frames[frame].ref = 1
    return

  if len(sim.page_table) < sim.num_frames:
    frame_index = len(sim.page_table)
  else:
    while True:
      hand = sim.clock_hand
      meta = sim.frames[hand]

      if getattr(meta, "ref", 0) == 0:
        del sim.page_table[meta.page_id]
        frame_index = hand
        break
      else:
        meta.ref = 0
        sim.clock_hand = (hand + 1) % sim.num_frames

    sim.clock_hand = (sim.clock_hand + 1) % sim.num_frames

  sim.page_table[page_id] = frame_index
  sim.frames[frame_index] = PageMeta(
    page_id=page_id,
    load_time=sim.time,
    last_access=sim.time,
    access_count=1,
    ref=1,
  )


# ---------- Sequential hybrid (LRU → MRU) ----------

RECENT_WINDOW_FOR_SEQ = 96  # how far back in time a page counts as "recent"


def seq_hybrid_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Hybrid policy for sequential / strided patterns.

  Idea:
    - Identify pages that have been accessed recently (within RECENT_WINDOW_FOR_SEQ).
    - Count how many frames currently hold such "recent" pages.
    - If recent pages occupy less than half of the frames:
        behave like LRU (evict the least recently used page) so the
        new sequential pages can fill the cache and old pages are flushed.
    - Once recent pages occupy at least half of the frames:
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


# ----- Preserve-cache policy (no eviction) -----


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


# ----- ML policy -----


def ml_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  ML-driven eviction policy.

  - Delegates pattern recognition and bucket selection to the global
    PatternController (ML_CONTROLLER).
  - For random regions (bucket B), returns -1 to signal that nothing should
    be evicted and the cache contents should remain unchanged.
  - Otherwise, delegates to the currently chosen concrete policy.
  """
  if ML_CONTROLLER is None:
    return lru_policy(frames, page_table, meta, time)

  bucket_name = ML_CONTROLLER.current_bucket_name()

  # If not enough history yet, fall back to LRU.
  if bucket_name is None:
    return lru_policy(frames, page_table, meta, time)

  # Bucket B: truly random region → skip eviction entirely.
  if bucket_name == "B":
    return -1

  policy_name = ML_CONTROLLER.current_policy_name()
  policy_fn = _POLICY_REGISTRY.get(policy_name, lru_policy)

  # Guard against accidental recursion
  if policy_fn is ml_policy:
    policy_fn = lru_policy

  return policy_fn(frames, page_table, meta, time)


# ----- Registry -----


_POLICY_REGISTRY: Dict[str, EvictionPolicyFn] = {
  "fifo": fifo_policy,
  "clock": clock_policy,
  "lru": lru_policy,
  "mru": mru_policy,
  "random": random_policy,
  "lfu": lfu_policy,
  "lhd": lhd_policy,
  "2q": twoq_policy,
  "seq_hybrid": seq_hybrid_policy,
  "seq_hybrid_policy": seq_hybrid_policy,  # alias to match PatternController mapping
  "preserve_cache": preserve_cache_policy,
  "ml": ml_policy,
}


def get_policy(name: str) -> EvictionPolicyFn:
  key = name.lower()
  if key not in _POLICY_REGISTRY:
    raise ValueError(
      f"Unknown policy '{name}'. Available: {list(_POLICY_REGISTRY)}"
    )
  return _POLICY_REGISTRY[key]
