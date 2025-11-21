# OS Concepts x ML Practice Project

## Initial Ideas:

1. ML Scheduler
2. ML Page Evictor Policy <--
3. ML Suspicious Process Classifier
4. ML Thread Predictor 

## Selected Project: ML Page Evictor

### Reasoning

1. No necessity modify a kernel
2. More easily implementable simulator
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

3. Define the ML problem, goal, and plan for training

4. Train the model(s)
  1. 

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

### Classifying Memory Access Patterns - Multiple Sources 

1. Sequential / scan – contiguous page references
2. Strided / structured – regular non-contiguous spatial accesses
3. Random / irregular – non-local, unpredictable page jumps
4. High-locality / hot-set – strong temporal locality within a small working set
5. Looping / cyclic – pages revisited in regular cycles (repeated strided/sequential)
6. Phase-shift / working-set change – workloads that move between pattern regimes / working sets
7. Moderate locality – partial working-set coverage between hot-set and random 
8. Mixed / multimodal – combinations of the above within one trace

Buckets (For which to define a best cache policy):

A: 1, 2
B: 3
C: 4
D: 5
E: 6, 7, 8

## 2. Creating a simulator

### Python Chosen

1. Much more simple to create code for
2. ML integration straightforward (PyTorch, TensorFlow, Scikit-Learn).
3. Copious and accessible visualization tools (Matplotlib, Seaborn) help analyze traces, patterns, and miss-rate curves.

### Simulator.py

Basic simulator of an operating system’s page-eviction behavior.

The simulator is completely policy-agnostic, acting purely on specificed policy.
It maintains all state needed to model demand paging, including:

1. Frame array representing physical memory
2. PageMeta: load time, last access time, access count
3. Page table mapping page IDs to frame indices
4. Counters for hits, faults, and total references

usage: python simulator.py --cacheframes <N> --policy <name> --trace <file> [--verbose]

A SimulationResult dataclass summarizes the run, reporting hit rate, fault rate, and the final content of the cache.

#### Methods:

access(page_id):
Simulates referencing a single page

run(trace):
Executes the full sequence of page references

main:
The main() function provides a simple CLI interface for running the simulator, and...

1. Parses command-line arguments (--frames, --policy, --trace, --verbose)
2. Loads the requested page trace
3. Retrieves the selected eviction policy
4. Instantiates a Simulator
5. Runs the simulation
6. Prints a summary of results
7. (If Verbose): Displays the final cache state and cache state at each access

### Policies.py

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

### Traces.py

Provides a set of pluggable trace generators that model a variety of memory-access patterns. These routines are used to benchmark cache-replacement policies, evaluate ML classifiers, and explore locality behavior. Each generator returns a list of integer page IDs suitable for direct input to the simulator.

#### Included Synthetic Trace Generators

1. **loop_small / loop_large**  
   Cyclic access patterns over small or large fixed working sets.  
   Models tight loops and repeated computations with strong temporal locality.

2. **locality_shift**  
   Splits the trace into several segments, each with a distinct, non-overlapping working set.  
   Models abrupt phase changes and working-set replacement.

3. **moving_working_set**  
   Generates a simple shifting window. Each phase selects a new contiguous region and draws uniformly from it.  
   Captures coarse working-set movement but without realistic overlap or temporal correlation.

4. **shifting_working_set**  
   Produces a more realistic drifting working set.  
   Each phase shifts the base address by a small amount, maintaining continuity and gradual locality evolution.  
   Models sliding-window access patterns and iterative workloads where the hot region moves over time.

5. **mixed_random**  
   Implements a hot–cold access model. A small hot subset receives most accesses; the remainder are drawn from a larger cold set.  
   Useful for testing eviction policies under skewed distributions.

6. **random**  
   Uniform random sampling across the full page range.  
   Provides a locality-free baseline and adversarial workload.

#### External Trace Support

If the provided name does not match a predefined trace, it is treated as a file path.  
Trace files may contain:

- One page ID per line, or  
- Space-separated integers

Empty lines and `#` comments are ignored.

#### Predefined Trace Names



## 3. Defining an ML Goal

### Goal: Bélády's Optimal Solution

Though not gauranteeable in reality, my goal is to replace the page that will be accessed furthest in the future, which is the optimal policy (Bélády’s) optimal replacement, using only information available at runtime (past accesses), and compare it to some of the policies built into the simulator.

To strive for Béláday, I plan on training a model to use a similar LPR strategy as described in the research paper cited above. I will train a mof=del that can recognize types of access patterns. This taxonomy will map to the optimal eviction strategy to use. When the model realizes new patterns in the incoming page access stream, it will change the eviction policy as necessary.

I plan on looking into ways I can improve the model beyond simple reactionary policy changes, such as building associations between certain pages as to incentivise keeping clustered pages in the cache even if a policy change hasn't been declared by the model. This might complicate things at a lower level, so this addition is still a hypothetical.

## 4. Training the Model(s)

### Building the Model

## ML Access Pattern Classifier

This project uses a small neural network to recognize different memory access patterns online and switch page replacement policies accordingly.

### Access pattern buckets (A–F)

I group windows of recent page references into six high-level buckets:

- **A – Scan / Stride (no reuse)**  
  Mostly sequential or strided forward accesses with almost no repeats.

- **B – Random**  
  Uniform or near-uniform random reads with minimal spatial or temporal locality.

- **C – Tight hot set**  
  Strong temporal locality within a small working set that fits well in cache.

- **D – Looping / cyclic**  
  Repeated traversal of a working set in a loop, possibly larger than the cache.

- **E – Dynamic hot / phase shift**  
  Hot pages embedded in a larger background plus phase changes where the hot region shifts over time.

- **F – Mixed / transition**  
  Ambiguous windows that contain a combination of patterns (e.g., during a change in behavior). When F is detected, the controller keeps the previous stable policy instead of switching immediately.

Each bucket is mapped to an “optimal” concrete policy (MRU, random, 2Q, LHD, etc.) that is better suited for that behavior.

### Feature vector

For each sliding window of recent page IDs, I compute an 8-dimensional feature vector:

### Feature Vector Definition

In order to start the model, I must determine an informative feature vector for my model to interpret/draw conclusions from during training. This feature vector will incorporate 8 important features of a window of past memory accesses, such as:

#### **1. `unique_ratio`**
Measures the proportion of unique pages in the window.  
High values indicate streaming, random, or shifting behavior; low values indicate strong temporal locality.

#### **2. `repeat_ratio`**
Defined as `1 − unique_ratio`.  
Quantifies how heavily the workload reuses the same pages.

#### **3. `sequential_frac`**
Fraction of adjacent accesses where `page[i+1] == page[i] + 1`.  
Identifies sequential scanning behavior.

#### **4. `stride_small_frac`**
Fraction of adjacent accesses with a small stride (e.g., `|diff| ≤ 4`).  
Captures structured spatial locality, including strided or near-linear patterns.

#### **5. `reuse_mean`**
Mean reuse distance for repeated page accesses.  
Lower values indicate strong temporal locality (hot sets, loops); higher values indicate streaming or random behavior.

#### **6. `reuse_small_frac`**
Fraction of reuse distances ≤ 10.  
Measures short-term temporal locality, distinguishing tight hot sets from larger loops or random behavior.

#### **7. `entropy`**
Entropy of the page-ID distribution in the window.  
High entropy indicates scattered or random access; low entropy indicates concentrated hot sets.

#### **8. `max_run_len`**
Length of the longest consecutive run of the same page.  
Identifies extreme immediate locality (e.g., heavy reuse of one page).

These features summarize spatial structure, temporal locality, and randomness in a compact way that is easy for my small network to learn from and make in-the-moment decisions on.

### Model architecture and training

The classifier is a simple feed-forward network implemented in PyTorch:

- Input: 8-dimensional feature vector  
- Layers: 32 → 32 hidden units with ReLU activations  
- Output: 6 logits (one per bucket A–F)  
- Loss: cross-entropy  
- Optimizer: Adam  
- Training data: ~6k windows generated from synthetic traces that simulate each bucket plus mixed windows for F.

The trained weights are stored in `access_pattern_classifier.pt`.

### Online PatternController

At runtime, the `PatternController`:

1. Maintains a sliding buffer of the most recent 50 page references.  
2. Computes the feature vector for the current window.  
3. Runs the classifier to predict the most likely bucket (A–F).  
4. Smooths predictions over the last few windows to avoid flapping.  
5. Maps the bucket to a concrete eviction policy:
   - A → MRU  
   - B → random  
   - C → 2Q  
   - D → 2Q  
   - E → LHD  
   - F → keep the last stable policy

The simulator calls `controller.observe(page_id)` on every reference, and the ML-driven `ml` policy queries `controller.current_policy_name()` whenever it needs to evict a page. This allows the cache to adapt dynamically to changes in the access pattern instead of committing to a single static replacement strategy.

### Some Model Changes!

After training the momdel and running over test data, i found taht a window of 100 felt a little bit 

1. Cutting Losses on Buckets A
   - A → LRU until 50% of cache has been replaced, !!!!!!!!  
   - B → NONE
   - C → 2Q
   - D → 2Q
   - E → LHD  
   - F → keep the last stable policy

The simulator calls `controller.observe(page_id)` on every reference, and the ML-driven `ml` policy queries `controller.current_policy_name()` whenever it needs to evict a page. This allows the cache to adapt dynamically to changes in the access pattern instead of committing to a single static replacement strategy.


## Further Improvements Discussion

1. Make use of more page metadata
  1. origin of page/type of segment read from
  2. (Increases feature data for potential model training)

2. Adaptive ML Data:
  1. Storage Size Recognition: In the simulator, there isn't a set storage size. Having a defined storage size could help allude to the type of memory accesses that are occuring. For instance, (1, 49, 24, 18) looks a lot more like a working set (rather than random calls) if your storage size is 1M pages than if your storage size is 50.
  2. Page Clustering/Association: Currently, the model only makes decisions based on teh current window, however, if was able to recognize cretain groups of pages together, it may infer a change in the access pattern earlier than it woueld under the curent model.
  3. ML Pattern Window Size: Even after research, I haven't quite found a consistent hot-pages/working set size (relative to the cache). I have found that cache size plays a vital role in the importance of policy switches, so being able to decide a reasonable cache size relative to both the storage size and typical hot pages size would allow me to choose a golidlocks window size, allowing the model to have enough data before a change is made, as well as not have too small of a scope.
  4. Pattern Frequency Targets: In training the model, I have 6 labeled memory access patterns. Though I am not fully sure how the model treats these patterns in terms of frequency in practice, I think it is reasonable to assume that the model COULD work better if I had more data to inform the model on the typical frequency of each of these patterns. For example, if the model is told that 80% of the time looping reads occur and 20% of the time random reads occur, the model might aim for such a split when it comes to the policy decisions it is making. Currently, though, I provide no data on the relative frequency of the 6 classifications I've defined.
