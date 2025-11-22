from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Iterable, Tuple

from ml.pattern_controller import PatternController
import policies
import traces


# Data structures 

@dataclass
class PageMeta:
  page_id: int
  load_time: int
  last_access_time: int
  access_count: int = 0


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


# Simulator core 

class Simulator:
  """
  Simple demand-paged virtual memory simulator.
  """

  def __init__(
    self,
    num_frames: int,
    policy_name: str,
    policy_fn: EvictionPolicyFn,
    controller=None,
    verbose: bool = False,
  ):
    self.num_frames = num_frames
    self.policy_name = policy_name
    self.policy_fn = policy_fn
    self.controller = controller
    self.verbose = verbose

    self.frames: List[Optional[int]] = [None] * num_frames
    self.page_table: Dict[int, int] = {}
    self.meta: Dict[int, PageMeta] = {}

    self.time: int = 0
    self.num_references: int = 0
    self.num_hits: int = 0
    self.num_faults: int = 0


  # Internal helpers 

  def _is_in_memory(self, page_id: int) -> Tuple[bool, Optional[int]]:
    frame_idx = self.page_table.get(page_id)
    return (frame_idx is not None, frame_idx)

  def _find_free_frame(self) -> Optional[int]:
    for i, page in enumerate(self.frames):
      if page is None:
        return i
    return None

  def _load_page_into_frame(self, page_id: int, frame_idx: int) -> None:
    self.frames[frame_idx] = page_id
    self.page_table[page_id] = frame_idx

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
    if page_id is not None:
      del self.page_table[page_id]
    self.frames[frame_idx] = None


  # Public API 

  def access(self, page_id: int) -> None:
    self.num_references += 1
    self.time += 1

    if self.controller is not None:
      self.controller.observe(page_id)

      if self.verbose:
        bucket = self.controller.current_bucket_name()
        policy = self.controller.current_policy_name()
        if bucket is not None:
          print(f"[ML] bucket={bucket}  policy={policy}")

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
        self.meta[page_id].access_count += 1

      else:
        victim_idx = self.policy_fn(
          self.frames,
          self.page_table,
          self.meta,
          self.time,
        )

        #-1: do not evict anything.
        if victim_idx == -1:
          if self.verbose:
            print("[ML] random region detected → skipping eviction; cache unchanged.")
          # num_faults was already incremented; the page is simply not loaded.
          return

        if not (0 <= victim_idx < self.num_frames):
          raise RuntimeError(
            f"Policy {self.policy_name} returned invalid frame index {victim_idx}"
          )

        self._evict_page_from_frame(victim_idx)
        self._load_page_into_frame(page_id, victim_idx)
        meta = self.meta[page_id]
        meta.last_access_time = self.time
        meta.access_count += 1

    # iterative cache print:
    #if self.verbose:
    #  print(f"[t={self.time}] Access {page_id} "
    #        f"({'HIT' if in_mem else 'FAULT'})")
    #  print("  Cache state:")
    #  for idx, p in enumerate(self.frames):
    #    print(f"    Frame {idx}: {p}")
    #  print("-" * 40)

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


# CLI 

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Page replacement simulator."
  )
  parser.add_argument("--frames", "-f", type=int, default=30)
  parser.add_argument("--policy", "-p", type=str, required=True)
  parser.add_argument("--trace", "-t", type=str, required=True)
  parser.add_argument("--verbose", "-v", action="store_true")
  return parser.parse_args()


def load_trace(trace_arg: str) -> List[int]:
  if os.path.isfile(trace_arg):
    with open(trace_arg, "r") as f:
      return [int(line.strip()) for line in f if line.strip()]

  return traces.get_trace(trace_arg)



def main() -> None:
  args = parse_args()

  policy_fn = policies.get_policy(args.policy)
  trace = load_trace(args.trace)

  controller: PatternController | None = None
  if args.policy.lower() == "ml":
    controller = PatternController(
      model_path="access_pattern_classifier.pt",
      window_size=192,
      history_len=8,
    )
    policies.set_ml_controller(controller)

  sim = Simulator(
    num_frames=args.frames,
    policy_name=args.policy,
    policy_fn=policy_fn,
    controller=controller,
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
    print("Final cache state:")
    for idx, page_id in enumerate(sim.frames):
      print(f"  Frame {idx}: {page_id}")


if __name__ == "__main__":
  main()
