from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Iterable, Tuple

import policies
import traces


# ----- Data structures -----

@dataclass
class PageMeta:
  """Metadata about a page needed for policies and ML."""
  page_id: int
  load_time: int          # reference index when loaded into memory
  last_access_time: int   # last reference index that touched this page
  access_count: int = 0   # how many times this page has been accessed


@dataclass
class SimulationResult:
  num_frames: int
  policy_name: str
  num_references: int
  num_hits: int
  num_faults: int

  def fault_rate(self) -> float:
    return self.num_faults / self.num_references if self.num_references else 0.0

  def hit_rate(self) -> float:
    return self.num_hits / self.num_references if self.num_references else 0.0


EvictionPolicyFn = Callable[
  [List[Optional[int]], Dict[int, int], Dict[int, PageMeta], int],
  int,
]


# ----- Simulator core -----

class Simulator:
  """
  Simple demand-paged virtual memory simulator.

  - Frames are represented as a fixed-size list of page IDs (or None if empty).
  - page_table maps page_id -> frame_index.
  - meta holds per-page metadata (load time, last access time, access count).
  """

  def __init__(
    self,
    num_frames: int,
    policy_name: str,
    policy_fn: EvictionPolicyFn,
    verbose: bool = False,
  ) -> None:
    if num_frames <= 0:
      raise ValueError("num_frames must be positive")

    self.num_frames: int = num_frames
    self.policy_name: str = policy_name
    self.policy_fn: EvictionPolicyFn = policy_fn
    self.verbose: bool = verbose

    self.frames: List[Optional[int]] = [None] * num_frames
    self.page_table: Dict[int, int] = {}      # page_id -> frame index
    self.meta: Dict[int, PageMeta] = {}       # page_id -> metadata

    self.time: int = 0                        # reference counter
    self.num_references: int = 0
    self.num_hits: int = 0
    self.num_faults: int = 0

  # ----- Internal helpers -----

  def _is_in_memory(self, page_id: int) -> Tuple[bool, Optional[int]]:
    frame_idx = self.page_table.get(page_id)
    if frame_idx is None:
      return False, None
    return True, frame_idx

  def _find_free_frame(self) -> Optional[int]:
    for i, p in enumerate(self.frames):
      if p is None:
        return i
    return None

  def _load_page_into_frame(self, page_id: int, frame_idx: int) -> None:
    self.frames[frame_idx] = page_id
    self.page_table[page_id] = frame_idx
    # initialize metadata if not present
    if page_id not in self.meta:
      self.meta[page_id] = PageMeta(
        page_id=page_id,
        load_time=self.time,
        last_access_time=self.time,
        access_count=0,
      )
    else:
      m = self.meta[page_id]
      m.load_time = self.time
      m.last_access_time = self.time

  def _evict_page_from_frame(self, frame_idx: int) -> None:
    page_id = self.frames[frame_idx]
    if page_id is None:
      return
    del self.page_table[page_id]
    self.frames[frame_idx] = None

  # ----- Public API -----

  def access(self, page_id: int) -> None:
    self.num_references += 1
    self.time += 1

    in_mem, frame_idx = self._is_in_memory(page_id)

    if in_mem:
      self.num_hits += 1
      meta = self.meta[page_id]
      meta.last_access_time = self.time
      meta.access_count += 1
    else:
      self.num_faults += 1
      free_frame = self._find_free_frame()

      if free_frame is not None:
        self._load_page_into_frame(page_id, free_frame)
        meta = self.meta[page_id]
        meta.last_access_time = self.time
        meta.access_count += 1
      else:
        victim_idx = self.policy_fn(
          self.frames,
          self.page_table,
          self.meta,
          self.time,
        )
        if not (0 <= victim_idx < self.num_frames):
          raise RuntimeError(
            f"Policy {self.policy_name} returned invalid frame index {victim_idx}"
          )
        self._evict_page_from_frame(victim_idx)
        self._load_page_into_frame(page_id, victim_idx)
        meta = self.meta[page_id]
        meta.last_access_time = self.time
        meta.access_count += 1

    if self.verbose:
      print(f"[t={self.time}] Accessed page {page_id} "
            f"({'HIT' if in_mem else 'FAULT'})")
      print("  Cache state (frame_index: page_id):")
      for idx, p in enumerate(self.frames):
        print(f"    Frame {idx}: {p}")
      print("-" * 40)

  def run(self, trace: Iterable[int]) -> SimulationResult:
    for page_id in trace:
      self.access(page_id)

    return SimulationResult(
      num_frames=self.num_frames,
      policy_name=self.policy_name,
      num_references=self.num_references,
      num_hits=self.num_hits,
      num_faults=self.num_faults,
    )


# ----- CLI -----

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Page replacement simulator with pluggable policies."
  )
  parser.add_argument(
    "--frames",
    "-f",
    type=int,
    required=True,
    help="Number of physical frames.",
  )
  parser.add_argument(
    "--policy",
    "-p",
    type=str,
    required=True,
    help="Policy name (e.g., fifo, lru, mru, 2q, random, ml).",
  )
  parser.add_argument(
    "--trace",
    "-t",
    type=str,
    required=True,
    help="Trace name or file path (depends on traces.py).",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print cache contents after each access.",
  )
  return parser.parse_args()


def load_trace(trace_arg: str) -> List[int]:
  return traces.get_trace(trace_arg)


def main() -> None:
  args = parse_args()

  policy_fn = policies.get_policy(args.policy)

  trace = load_trace(args.trace)

  sim = Simulator(
    num_frames=args.frames,
    policy_name=args.policy,
    policy_fn=policy_fn,
    verbose=args.verbose,
  )
  result = sim.run(trace)

  print(f"Policy:      {result.policy_name}")
  print(f"Frames:      {result.num_frames}")
  print(f"References:  {result.num_references}")
  print(f"Page hits:   {result.num_hits}")
  print(f"Page faults: {result.num_faults}")
  print(f"Hit rate:    {result.hit_rate():.4f}")
  print(f"Fault rate:  {result.fault_rate():.4f}")

  if args.verbose:
    print("Final cache state (frame_index: page_id):")
    for idx, page_id in enumerate(sim.frames):
      print(f"  Frame {idx}: {page_id}")

if __name__ == "__main__":
  main()
