from __future__ import annotations
import random
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
  from simulator import PageMeta

EvictionPolicyFn = Callable[
  [List[Optional[int]], Dict[int, int], Dict[int, "PageMeta"], int],
  int,
]


# ---------- FIFO ----------

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


# ---------- LRU ----------

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


# ---------- MRU ----------

def mru_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  """
  Evict the most recently used page.
  Uses meta[page_id].last_access_time.

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


# ---------- Random ----------

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


# ---------- LFU ----------

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


# ---------- LHD ----------

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


# ---------- 2Q-style ----------

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

  victim_idx: Optional[int] = None

  if once_seen:
    victim_idx = min(once_seen, key=lambda pair: pair[1])[0]
  elif multi_hit:
    victim_idx = min(multi_hit, key=lambda pair: pair[1])[0]
  else:
    victim_idx = 0

  return victim_idx


# ---------- ML stub ----------

def ml_policy(
  frames: List[Optional[int]],
  page_table: Dict[int, int],
  meta: Dict[int, "PageMeta"],
  time: int,
) -> int:
  return lru_policy(frames, page_table, meta, time)


# ---------- Registry ----------

_POLICY_REGISTRY: Dict[str, EvictionPolicyFn] = {
  "fifo": fifo_policy,
  "lru": lru_policy,
  "mru": mru_policy,
  "random": random_policy,
  "lfu": lfu_policy,
  "lhd": lhd_policy,
  "2q": twoq_policy,
  "ml": ml_policy,
}


def get_policy(name: str) -> EvictionPolicyFn:
  key = name.lower()
  if key not in _POLICY_REGISTRY:
    raise ValueError(
      f"Unknown policy '{name}'. "
      f"Available: {list(_POLICY_REGISTRY)}"
    )
  return _POLICY_REGISTRY[key]
