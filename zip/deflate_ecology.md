# Deflate Ecology Design

## Purpose

This directory sketches a compression research direction that applies the project's puzzle ecology and composition-gene ideas to deflate-style compression.

The core idea is:

```text
corpus ecology discovers hard compression borders;
regression scores parse/table candidates;
composition genomes describe compressor operators;
deflate streams act as executable reconstruction programs.
```

This is not a replacement for the deflate format. The goal is to evolve and learn better choices inside the format's legal decision space.

## Why deflate fits the ecology model

Deflate combines two interacting systems:

1. **LZ77 parsing**
   - choose literal bytes or length-distance backreferences;
   - discover reusable substrings;
   - decide whether to take a short match now or preserve context for a later match.

2. **Huffman coding**
   - build literal/length tables;
   - build distance tables;
   - choose static or dynamic blocks;
   - pay table overhead only when the block earns it back.

Those choices form a living decision surface. The interesting regions are borders:

```text
literal vs match
static table vs dynamic table
new block vs same block
speed vs ratio
local reuse vs global reuse
random-looking data vs weak structure
```

Puzzle ecology becomes **corpus ecology**: a population of data windows that exposes where compression decisions are uncertain, brittle, or high-value.

## Main analogy

| SAT ecology concept | Deflate ecology concept |
| --- | --- |
| Puzzle instance | Corpus window / byte slice |
| Puzzle species | Text, logs, JSON, source, binary, random, mixed payload |
| Puzzle traits | Entropy, repetition, match histogram, distance histogram |
| Solver composition gene | Compression operator gene |
| Composition genome | Compressor pipeline / parse-table strategy |
| Border puzzle | Compression window near a decision boundary |
| Fitness | Size, speed, validity, stability, generalization |
| Mutation | Retune parser/table/block policy |

## Deflate as executable reconstruction

A deflate stream can be treated as a restricted program for a decompressor.

Its instructions are roughly:

```text
emit literal byte
copy length bytes from previous distance
switch block mode
use this Huffman table
end block
```

The decompressor executes these instructions over an output tape. A backreference is a constrained function call into already-produced output:

```text
distance = pointer to reusable prior structure
length   = amount of structure to reuse
```

This does not make raw deflate a general-purpose virtual machine. It does make deflate a small executable language for reconstruction. The richer computational loop appears when we add ecology and regression:

```text
candidate deflate program
  -> decompressed output
  -> validity and performance tests
  -> regression loss
  -> mutation / selection
  -> next candidate program
```

## Compression genes

A first composition genome for deflate could include these gene families.

### Corpus trait genes

```text
byte_histogram
byte_entropy
literal_skew
n_gram_repetition
match_length_histogram
distance_histogram
stationarity_score
randomness_proxy
```

### Parser genes

```text
match_finder
literal_vs_match_scorer
lazy_match_depth
short_match_penalty
long_match_reward
distance_penalty
parse_regret_estimator
```

### Table genes

```text
static_table_candidate
dynamic_table_candidate
smoothed_dynamic_table
neighbor_inherited_table
cluster_prototype_table
literal_length_table_builder
distance_table_builder
table_overhead_estimator
```

### Block genes

```text
block_splitter
block_merge_scorer
reset_policy
flush_policy
static_vs_dynamic_selector
stored_block_escape
```

### Fitness genes

```text
compressed_size
decode_speed_proxy
encode_speed_proxy
table_overhead
valid_roundtrip
regret_vs_oracle
stability_across_neighbor_windows
generalization_to_similar_corpus
```

## Regression loop

The regression problem is selection among candidates.

For each corpus window:

```text
features(window, candidate_strategy) -> predicted_cost
```

Where cost can combine:

```text
compressed_size
+ encode_time_weight * encode_time
+ decode_time_weight * decode_time
+ instability_penalty
+ invalidity_penalty
```

The loop:

```text
corpus window
  -> extract traits
  -> generate parse/table/block candidates
  -> score candidates with regression
  -> encode with selected candidate
  -> verify roundtrip
  -> measure actual cost
  -> compute loss
  -> update scorer
```

The first scorer can be a transparent weighted rule model. Later it can become a linear regressor, pairwise ranker, or small model trained from benchmark rows.

## Active borders on compression genes

A gene is on the border when its decision has high uncertainty or high regret.

Examples:

| Border signal | Pressured gene |
| --- | --- |
| Dynamic-table overhead exceeds compression gain | block splitter / table selector |
| Static table nearly matches dynamic table | static-vs-dynamic selector |
| Many literal/match near-ties | literal-vs-match scorer |
| Long matches appear after greedy short matches | lazy parser |
| Distance symbols dominate cost | distance table builder |
| Adjacent windows prefer different tables | block reset policy |
| Predicted cost misses actual cost | regression feature/scorer gene |
| Roundtrip fails | emitter/validator gene |

The ecology should not search only for easiest compression wins. It should preserve border windows that teach the compressor where its reuse model is weak.

## Minimal implementation plan

### Phase 1: Passive corpus ecology

Build metrics over byte windows:

```text
window_size
byte_entropy
literal_skew
repeat_density
candidate_match_count
match_length_mean
match_length_peak
distance_mean
distance_peak
static_deflate_size
dynamic_deflate_size
stored_size
static_dynamic_gap
table_overhead_estimate
compression_border_score
corpus_niche
```

No learned behavior yet. Just annotate benchmark rows.

### Phase 2: Candidate strategy scoring

Generate several legal deflate strategies per window:

```text
stored
static_fast
dynamic_default
dynamic_smoothed
block_split_aggressive
block_split_conservative
lazy_parse_low
lazy_parse_high
```

Measure actual output size and fit a scorer that predicts which strategy should win.

### Phase 3: Composition genome

Represent the compressor as operators:

```text
bytes -> corpus_traits
bytes + corpus_traits -> match_candidates
match_candidates + traits -> parse
parse + traits -> symbol_counts
symbol_counts + traits -> huffman_tables
parse + huffman_tables -> deflate_stream
deflate_stream -> decoded_bytes
bytes + decoded_bytes -> roundtrip_valid
```

Then expose operator genes, active borders, and mutation candidates using the same pattern as the SAT ecology work.

### Phase 4: Executable mutation replay

For selected border genes, run bounded replay mutations:

```text
lower static-vs-dynamic threshold
increase lazy depth
smooth table frequencies
split block earlier
merge neighboring block
penalize long distances
prefer stored block for high entropy
```

Compare baseline and mutant output on the same window.

## Safety and correctness constraints

Every generated stream must pass roundtrip validation:

```text
decompress(compress(input)) == input
```

Hard constraints should be validators, not learned preferences:

```text
valid Huffman code lengths
legal distance values
legal match lengths
block format correctness
checksum/container correctness when ZIP wrapper is used
```

Regression may select among valid candidates, but it should not be allowed to bypass format validators.

## Initial success criteria

A useful first prototype would show:

1. Corpus windows receive stable ecology traits.
2. Border windows are identified where static/dynamic or literal/match choices are close.
3. Candidate strategies produce valid roundtrips.
4. A simple scorer predicts winners better than a fixed default on held-out windows.
5. Mutant replays improve some border windows without increasing invalid output.

## Bootstrap experimentation layer

The SAT ecology code is already stabilizing a useful pattern for built-in experimentation:

```text
observe motifs
  -> infer landscape needs
  -> translate motifs into effect providers
  -> compose a bootstrap plan
  -> expose missing effects as pressure
  -> choose bounded experiments
```

For deflate ecology, we can reuse that shape directly. The point is not to rank every compression trick globally. The point is to ask:

```text
what does this compression climate need right now,
and which observed motifs can provide it?
```

### Deflate motif observations

A compression run can emit ordered motifs just like solver operator traces. Examples:

```text
literal_run -> short_match
short_match -> long_match
static_block -> dynamic_block
dynamic_block -> stored_block
block_split -> inherited_table
high_entropy_window -> stored_escape
match_burst -> distance_table_shift
```

Each motif should stay descriptive at first:

```text
source
target
count
activation_rate
mean_size_delta
mean_decode_cost_delta
entropy_shift
persistence
role
```

Useful roles:

```text
reuse_discovery
scope_shift
language_stabilization
entropy_escape
table_repricing
front_crossing
```

### Bootstrap needs from compression climate

The climate layer should infer needs from corpus/window metrics:

| Climate signal | Bootstrap need |
| --- | --- |
| high static/dynamic near-tie | `table_choice_probe` |
| high adjacent-window cost volatility | `frontier_sensing` |
| repeated near-tie literal/match decisions | `parse_ambiguity_probe` |
| dynamic table wins after overhead | `local_language_building` |
| dynamic table loses by overhead | `scope_simplification` |
| long matches appear after greedy short matches | `lazy_parse_probe` |
| high entropy with weak repeats | `stored_escape_probe` |
| distance code cost dominates | `distance_topology_probe` |

This gives the experiment layer targets like:

```text
table_choice_probe
frontier_sensing
parse_ambiguity_probe
local_language_building
scope_simplification
lazy_parse_probe
stored_escape_probe
distance_topology_probe
```

### Motif effects as providers

Observed motifs can be translated into effect providers:

```text
reuse_discovery        -> parse_ambiguity_probe
scope_shift            -> frontier_sensing, scope_simplification
language_stabilization -> local_language_building
table_repricing        -> table_choice_probe, distance_topology_probe
entropy_escape         -> stored_escape_probe
front_crossing         -> frontier_sensing
```

The bootstrap composer can then ask for climate needs and produce an experiment plan.

Example:

```text
climate needs:
  frontier_sensing
  table_choice_probe

observed motif effects:
  block_split -> inherited_table provides frontier_sensing
  static_block -> dynamic_block provides table_choice_probe

bootstrap plan:
  block_split->inherited_table
  static_block->dynamic_block

missing:
  none
```

If a need is missing, that absence becomes experimental pressure rather than failure:

```text
missing local_language_building
  -> synthesize a candidate dynamic-table smoothing experiment
```

### Built-in experiment candidates

The first bounded experiments should be legal deflate strategy variants, not custom format changes:

```text
stored_vs_static_probe
static_vs_dynamic_probe
neighbor_table_inheritance_probe
smoothed_dynamic_table_probe
conservative_block_split_probe
aggressive_block_split_probe
low_lazy_depth_probe
high_lazy_depth_probe
short_distance_bias_probe
stored_escape_probe
```

Each experiment should record:

```text
experiment_name
source_need
source_motif
strategy_params
baseline_size
candidate_size
baseline_decode_cost
candidate_decode_cost
roundtrip_valid
size_delta
cost_delta
accepted
```

### Bootstrap selection rule

Start with inspectable rules:

```text
1. infer climate needs from passive corpus ecology;
2. compose observed motif effects toward those needs;
3. if a need is missing, pick the smallest synthetic probe for that need;
4. run only valid deflate candidates;
5. accept candidates that improve cost without breaking roundtrip;
6. preserve near-ties as future border windows.
```

This keeps experimentation local and bounded. The ecology generates hypotheses; validators enforce legal streams; regression later learns which bootstrap probes tend to pay off.

### Minimum viable bootstrap experiment

A first zip bootstrap prototype can be very small:

```text
input corpus windows
  -> passive corpus ecology metrics
  -> compare stored/static/dynamic zlib outputs
  -> infer climate needs
  -> compose motif bootstrap plan
  -> run one bounded candidate replay
  -> record baseline vs candidate size
```

The initial experiment does not need a custom deflate emitter. It can use available compression levels/strategies as stand-ins while the ecology and bootstrap machinery stabilize.

A useful first row shape:

```text
window_id
window_size
byte_entropy
repeat_density
stored_size
static_size
dynamic_size
static_dynamic_gap
compression_border_score
compression_weather_front_score
bootstrap_targets
bootstrap_plan
bootstrap_missing
bootstrap_action_hint
candidate_strategy
candidate_size
candidate_delta
roundtrip_valid
```

This mirrors the SAT bootstrap pattern while shifting the domain from solver motifs to compression motifs.

## Guiding phrase

```text
Deflate ecology evolves executable compression programs at the border between reuse and entropy.
```
