# Initial Ideation

## Selected Focus: ML-Based Page Evictor

### Reasoning

1. No Necessity modify a kernal
2. Decently easily implementable simulator
3. Recreateable standard policies
4. No worrying about race conditions!
5. Concrete and clear reward system for the model

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

## 2. Creating a simulator

### Python Chosen

1. Much more simple to create code for
2. ML integration made easier

### 3 Files:

## Policies

## 3. Defining an ML Goal 

### Bélády's Optimal Solution

Learn a page eviction policy that approximates Bélády’s optimal replacement, using only information available at runtime (past accesses), and compare it to FIFO, LRU, and Random.