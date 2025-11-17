# Initial Ideation - OS Concepts x ML Practice Project

1. ML Scheduler
2. ML Page Evictor Policy <--
3. ML Suspicious Process Classifier
4. ML Thread Predictor 

## Selected Project: ML Page Evictor

### Reasoning

1. No Necessity modify a kernal
2. Decently easily implementable simulator
3. Recreateable standard policies for comparison
4. No worrying about race conditions!
5. Concrete and clear reward system for the model's training

### Pre-research Ideas on Model

#### Live Learning: Multi-armed-bandit approach

1. Spend some time learning the manner in which pages are accessed and decide a pre-made policy based on what is optimal?

Could also...

2. Spend some time learning how certain pages are accessed--sequentially or randomly--and adapt policy as such. Could create clusters of page that are sequentially accessed/ routinely acessed together. Big fan of using this in model.

#### Pre-trained algorithm?

Seems much less optimal... not adaptive? not quite sure how this would make good use of ML. Probably would cause less overhead though.

### Plan

1. Research Page Eviction (Refresher) and learn how ML is used with it

2. Create a simulator and add clear support for known policies, such as LRU, FIFO, Random

3. Define the ML problem

4. 

## 1. Research Findings:

### Learning-based page replacement scheme for efficient I/O processing - https://www.nature.com/articles/s41598-025-88736-4

#### EELRU - Early Eviction Least Recently Used

ML reaction to cache-size not being large enough for sequential reads to benefit from LRU

#### LPR - Learned Page Replacement

Multi-Armed-Bandit Approch: spend some time learning, then apply optimal learned policy for workload

LPR makes a policy selection after learning phase rather than being a new policy alone

The scheme achieved up to ~62% reduction in execution time in a real out-of-core graph processing case, showing strong performance gains.

### Enabling Customization of Operating System Memory Eviction Policies - https://er.ucu.edu.ua/server/api/core/bitstreams/8ec6969a-feee-4271-acfc-8d61cc1d820f/content

#### LHD - Least Hit Density

Removal based on usefulness relative to time spent in cache.

#### MRU - Most Recently Used

Self-explanatory, effective in the case of Sequential page reads followed by random followed by more sequential. A temporary MRU would keep those sequential pages 

#### LFU - Most Frequently Used

LHD without the division

#### LRU-2Q - 2 Queue Least Recently Used

LRU with a queue to track pages which have been accessed multiple times recently.

### Classifying Memory Access Patterns - Multiple Sites 

1. Sequential / scan – contiguous page references.
2. Strided / structured – regular non-contiguous spatial accesses


3. Random / irregular – non-local, unpredictable page jumps (madvise, FCM, AOIO, FlatFlash). 


High-locality / hot-set – strong temporal locality within a small working set (Denning, core working sets). 

Looping / cyclic – pages revisited in regular cycles (FCM; He et al.; AOIO). 

Phase-shift / working-set change – workloads that move between pattern regimes / working sets (AOIO mixed workloads + working-set theory). 

Moderate locality – partial working-set coverage between hot-set and random (implied by working-set and core-working-set analyses). 

Mixed / multimodal – combinations of the above within one trace (AOIO Multi4; DL I/O mixes). 

This gives you a source-backed graph of page access types that you can put directly into your write-up or slides.

## 2. Creating a simulator

### Python Chosen

1. Much more simple to create code for
2. ML integration made easier
3. ...                                                         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

### 3 Files:

## Simulator.py

# Overview

Basic simulator of an operating system’s page-eviction behavior.

The simulator is completely policy-agnostic, acting purely on specificed policy.
It maintains all state needed to model demand paging, including:

1. Frame array representing physical memory
2. PageMeta: load time, last access time, access count
3. Page table mapping page IDs to frame indices
4. Counters for hits, faults, and total references

usage: python simulator.py --cacheframes <N> --policy <name> --trace <file> [--verbose]

A SimulationResult dataclass summarizes the run, reporting hit rate, fault rate, and the final content of the cache.

# Methods:

access(page_id):
Simulates referencing a single page

run(trace):
Executes the full sequence of page references

main:
The main() function provides a simple CLI interface for running the simulator...

1. Parses command-line arguments (--frames, --policy, --trace, --verbose)
2. Loads the requested page trace
3. Retrieves the selected eviction policy
4. Instantiates a Simulator
5. Runs the simulation
6. Prints a summary of results
7. (If Verbose): Displays the final cache state

## Policies.py

Collection of pluggable page-replacement algorithms, each matching a single function signature:

Included policies:

1. FIFO – evict the earliest-loaded page
2. LRU – least recently used
3. MRU – most recently used
4. Random – uniform random eviction
5. LFU – least frequently used
6. LHD – least hit-density (continuous: hits / residency_time)
7. 2Q-style – protects multi-hit pages, ejects one-hit pages
8. ML (stub) – placeholder for the learned model                          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

Policies are registered in a central dictionary so the simulator can select them by name via CLI.

## Traces.py

Collection of pluggable cache access routines, each modeling a common access pattern.

Provides several synthetic workloads that mimic common memory-access patterns:

1. loop_small / loop_large – cyclic working sets
2. locality_shift – working set changes over time
3. mixed_random – hot vs cold pages with different probabilities

Also supports loading traces from external files.
All traces output simple lists of page IDs to feed into the simulator.

## 3. Defining an ML Goal

### Goal: Bélády's Optimal Solution

Though not gauranteeable in reality, my goal is to replace the page that will be accessed furthest in the future, which is the optimal policy [Bélády’s] optimal replacement, using only information available at runtime (past accesses), and compare it to FIFO, LRU, and Random.