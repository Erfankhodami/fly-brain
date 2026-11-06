# Fly-Brain: Project Understanding and Technical Reference

## Document scope and evidence

- **Prepared:** 2026-10-25.
- **Repository:** `fly-brain`.
- **Source branch reviewed:** `Erfan`.
- **Source commit reviewed:** `18b12c12ebb5021aa7e0abd05ce795da60dce68a` (`18b12c1`, “simulated the proejct for first time”).
- **Purpose:** explain what the project does, how its implementations work, what the saved artifacts establish, and where its current boundaries are.
- **Change scope:** this document is a new documentation artifact; the review did not change simulation code, environments, datasets, or existing documentation.

This document is based on the repository's source code, environment definitions, notebook sources, tracked result metadata, and read-only checks performed during this review. It distinguishes four kinds of evidence:

1. **Implemented:** a behavior is visible in the current source code.
2. **Recorded:** a saved CSV, manifest, or log describes a previous run.
3. **Verified during review:** a read-only command or validation actually succeeded in this session.
4. **Interpretation or recommendation:** an explanation or proposed improvement, not an implemented feature or a newly measured result.

The simulation backends were not rerun. The Python interpreter available to the review session did not have NumPy, pandas, PyArrow, Brian2, Brian2CUDA, PyTorch, or PyGeNN installed. That observation applies to this interpreter, not necessarily to other Conda environments or WSL installations on the machine.

Source references use `path:line` and refer to the reviewed commit. Counts describing local result files are a snapshot, not a permanent property of the repository.

## Contents

1. Executive understanding
2. Scientific model and terminology
3. Connectome data and identity mapping
4. Experiments and manipulation capabilities
5. Architecture and execution flow
6. Backend implementations
7. Trials, rounds, and benchmark grids
8. Output files and persistence rules
9. Timing methodology and interpretation
10. Accuracy and spike-output comparisons
11. Original-paper code and notebooks
12. Environments and setup dependencies
13. Current recorded results
14. Implementation differences and operational risks
15. Implemented versus not implemented
16. Suggested development priorities
17. Repository navigation guide
18. Verification performed
19. Licensing and provenance
20. Final assessment

## 1. Executive understanding

The project is a **connectome-based fruit-fly brain simulator with a multi-framework benchmarking and comparison layer**.

It constructs a recurrent spiking network from the wiring of an adult *Drosophila* brain. Selected neurons receive artificial stimulation, activity propagates through the network, and the program records the resulting spikes. The same broad model is implemented using several CPU/GPU simulation technologies so their execution costs and neural outputs can be compared.

The project therefore has two connected purposes:

- **Computational neuroscience:** investigate which downstream neurons respond to activation or silencing of selected neurons.
- **Simulation engineering:** compare setup cost, compilation cost, simulation throughput, and output agreement across frameworks.

The most useful short description is:

> Start with the fly's measured wiring, apply controlled neural inputs, simulate simplified electrical dynamics, and compare both the activity and the execution cost across implementations.

This is not currently a language model, an AI training pipeline, or a complete virtual animal. There is no implemented closed-loop body controller, physics integration, or feedback from a simulated environment into the brain.

The root README attributes the model to the Shiu et al. whole-brain leaky integrate-and-fire work. This review inspected the included implementation; it did not independently reproduce the upstream paper's biological validation. See `README.md:7` and `README.md:112`.

## 2. Scientific model and terminology

### 2.1 Basic terms

| Term | Meaning in this project |
|---|---|
| Connectome | The supplied graph of directed connections between neurons. |
| Neuron | A computational unit with a membrane-voltage state and threshold-based spiking. |
| Synaptic weight | A signed strength applied when an upstream neuron spikes. |
| Spike | A discrete firing event recorded with a neuron identity and time. |
| Spike train | The ordered firing times of one neuron in one trial. |
| Firing rate | Spike count divided by the relevant simulated duration, expressed in Hz. |
| Refractory period | A post-spike interval during which the canonical neuron dynamics are restricted. |
| Poisson activation | Random stimulation with a specified average event rate. |
| Trial | One simulation realization for a chosen experiment and duration. |
| Round | One repetition of the selected benchmark grid. |
| Backend | A framework-specific implementation and execution strategy. |

### 2.2 Canonical Brian2 dynamics

The Brian2 parameter definition is in `code/run_brian2_cuda.py:39`. Its neuron equations are:

```text
dv/dt = (v_0 - v + g) / t_mbr
dg/dt = -g / tau
```

Both equations are declared `unless refractory`. Here, `v` is membrane voltage and `g` is a voltage-valued synaptic-drive state. Despite conductance-related names elsewhere in the code, the Brian2 declaration gives `g` units of volts; it should not automatically be interpreted as a detailed physical conductance model.

An arriving recurrent spike adds its connection weight to `g`. A neuron spikes when its voltage exceeds the threshold, resets its voltage, and clears its synaptic-drive state. Synaptic effects arrive after the configured delay.

| Parameter | Value | Meaning |
|---|---:|---|
| Resting/initial voltage | −52 mV | Starting voltage and resting target. |
| Reset voltage | −52 mV | Voltage following a spike. |
| Spike threshold | −45 mV | Threshold in the canonical model. |
| Membrane time constant | 20 ms | Voltage evolution timescale. |
| Synaptic time constant | 5 ms | Decay timescale of synaptic drive. |
| Refractory period | 2.2 ms | Canonical non-stimulated-neuron refractory duration. |
| Recurrent synaptic delay | 1.8 ms | Delay before a downstream connection takes effect. |
| Synaptic scaling factor | 0.275 mV | Scale applied to the signed connectivity value. |
| Poisson scaling factor | 250 | Multiplier for external activation strength. |
| Explicit PyTorch/GeNN timestep | 0.1 ms | Discrete simulation step. |

Brian2 uses its `linear` integration method. The runner relies on Brian2's default timestep rather than explicitly setting it in this file; the other implementations describe their 0.1 ms step as matching that default.

### 2.3 External activation

In the Brian2 implementation, external activation changes membrane voltage directly, rather than entering the recurrent `g` state. The configured activation weight is `0.275 mV × 250 = 68.75 mV` per input event. Stimulated neurons have their refractory duration set to zero.

This is an artificial drive designed to induce spiking. It is not a simulation of every molecular mechanism of optogenetics or natural sensory transduction. A configured rate is an average stimulation rate, not a guarantee of exactly that many spikes in every second.

PyTorch and direct GeNN generate Bernoulli events at each timestep using a probability proportional to rate and timestep. Their random generators and update schedules are not automatically identical to Brian2's.

### 2.4 Scope of biological detail

The source uses broadly shared neuron parameters and fixed connectivity-derived weights. It does not implement neuron-specific biophysical parameter fitting, detailed dendritic compartments, synaptic plasticity, or a learning rule that changes the connectome during a benchmark.

“Whole-brain” describes the network's anatomical scope. It does not mean that every biological property of the brain is represented, or that every neuron must become active in an experiment.

## 3. Connectome data and identity mapping

### 3.1 Current dataset

The central paths are defined in `code/benchmark.py:44`:

- `data/2025_Completeness_783.csv`: current neuron list.
- `data/2025_Connectivity_783.parquet`: current connectivity.
- `data/archive/2023_Completeness_630.csv`: legacy neuron list.
- `data/archive/2023_Connectivity_630.parquet`: legacy connectivity.
- `data/sez_neurons.pickle`: neuron subset used by notebook analysis.

The current neuron CSV was read with the Python standard library. It contains **138,639 rows and 138,639 unique FlyWire IDs**, all marked `Completed=True`.

Its columns are an unnamed ID column and `Completed`; this file is principally an identity list, not a comprehensive table of neuron names, cell types, or positions.

The README describes the connectivity scale as roughly five million synapses. The connectivity Parquet's row count and complete schema were not independently decoded in this session because a Parquet reader was unavailable. The column contract below is grounded in what the runners actually read, not a fresh validation of every data row.

### 3.2 Connectivity columns consumed by the runners

| Column | Use |
|---|---|
| `Presynaptic_Index` | Integer index of the source neuron. |
| `Postsynaptic_Index` | Integer index of the target neuron. |
| `Excitatory x Connectivity` | Signed connection value before model-specific scaling. |

The code uses the sign of the supplied value to represent excitatory or inhibitory effects. It does not independently infer neurotransmitter identity at runtime.

### 3.3 FlyWire IDs versus internal indices

The neuron CSV determines ordering. The runners enumerate its IDs to create:

- `flyid2i`: FlyWire ID → internal array/neuron index.
- `i2flyid`: internal index → FlyWire ID.

Experiments are expressed in FlyWire IDs, while connectivity and simulation state use internal indices. Outputs include both identities so results can be interpreted biologically and checked computationally.

The CSV ordering and connectivity indices must remain compatible. Reordering the neuron CSV without regenerating compatible connectivity would change the meaning of indices.

FlyWire IDs should be preserved as integer identifiers, not treated as approximate floating-point values.

### 3.4 PyTorch sparse weights

`code/run_pytorch.py:285` loads or builds cached sparse matrices:

- `data/weight_coo.pkl`.
- `data/weight_csr.pkl`.

The matrix is constructed with rows corresponding to postsynaptic indices and columns to presynaptic indices. The forward pass multiplies the previous spikes by the transposed matrix to calculate recurrent input.

The caches improve reuse but have no source-data fingerprint or schema/version validation in the loader. Changing the dataset while keeping old caches can therefore reuse incompatible weights. Only locally generated, trusted pickle files should be loaded.

## 4. Experiments and manipulation capabilities

### 4.1 Built-in benchmark experiments

The current `EXPERIMENTS` registry is in `code/benchmark.py:57`.

| CLI key | Name | Activated neurons | Stimulation rate | Second group | Silenced group |
|---|---|---:|---:|---|---|
| `sugar` | Sugar GRNs | 21 | 200 Hz | Empty | Empty |
| `p9` | P9s forward walking | 2 | 100 Hz | Empty | Empty |

`sugar` is the default. Both built-in experiments use only the first activation group.

The two P9 IDs are preserved in the registry as:

- `720575940627652358`: P9 left.
- `720575940635872101`: P9 right.

“Forward walking” is the experiment's biological association. The runner records neural spikes; it does not produce a body trajectory or locomotion animation.

### 4.2 General experiment interface

The original Brian2 interface, `run_exp`, accepts:

- `neu_exc`: first activation group.
- `neu_exc2`: second activation group.
- `neu_slnc`: silencing group.
- `params`: neuron constants, activation rates, duration, and trial count.
- Data and output paths.
- Process count and overwrite behavior.

See `code/paper-phil-drosophila/model.py:295`.

The benchmark registry similarly carries the three neuron lists, but the main CLI does not expose arbitrary neuron IDs, arbitrary activation rates, or a general experiment-config file. New benchmark experiments currently require code/configuration work.

### 4.3 Capability differences

The presence of the same configuration keys does not mean all runners use them identically:

| Backend | First activation group | Second activation group | Silencing behavior visible in source |
|---|---|---|---|
| Brian2 CPU / Brian2CUDA | Implemented | Implemented, but second rate remains the default zero in the benchmark runner | Zero outgoing weights from selected neurons. |
| PyTorch | Implemented | Not consumed by current benchmark setup | `neu_slnc` is not consumed by current benchmark setup. |
| NEST GPU | Implemented | Constructs inputs, but sets their rate to zero | Suppresses spike emission through `slnc_on`. |
| Direct GeNN | Implemented | IDs are mapped and refractory values adjusted, but no second source population is created | Removes edges both to and from selected neurons. |
| Brian2GeNN | Implemented | Implemented, but second rate remains the default zero in the benchmark runner | Zero outgoing weights from selected neurons. |

References: `code/run_brian2_cuda.py:166`, `code/run_brian2_cuda.py:184`, `code/run_pytorch.py:368`, `code/run_nestgpu.py:183`, `code/run_genn.py:120`, and `code/run_brian2_genn.py:96`.

These differences do not establish that the recorded sugar experiment is invalid: its second and silenced groups are empty. They do mean arbitrary coactivation or silencing experiments cannot yet be assumed equivalent across frameworks.

## 5. Architecture and execution flow

### 5.1 Overall organization

```text
main.py
  |
  +-- Parse backend, experiment, duration, trial, and round options
  |
  v
code/benchmark.py
  |
  +-- Shared experiment registry, paths, logging, and CSV persistence
  +-- Repeat selected rounds
  +-- Dispatch selected backends sequentially
  |
  +--> run_brian2_cuda.py   --> Brian2 CPU or Brian2CUDA
  +--> run_pytorch.py      --> Custom PyTorch implementation
  +--> run_nestgpu.py      --> NEST GPU subprocess workers
  +--> run_genn.py         --> Direct PyGeNN CUDA implementation
  +--> run_brian2_genn.py  --> Brian2GeNN standalone implementation
  |
  v
Timing CSVs + optional spike Parquets + labeled manifests
  |
  v
Comparison scripts and notebook analysis
```

The repository is a script-oriented Python project, not a packaged service. `main.py:43` adds `code/` to the import path. There is no tracked Python packaging manifest, web server, frontend, or REST API.

### 5.2 Main CLI responsibilities

`main.py:52` handles:

- Backend selection.
- Duration and trial lists.
- Experiment selection.
- Paper-grid defaults.
- Round count and starting round.
- Output grouping labels.
- File logging.
- Timing-only execution without spike recording.

Without backend flags, it selects five main-environment backends in this order: Brian2 CPU, Brian2CUDA, PyTorch, NEST GPU, and GeNN. Brian2GeNN requires explicit selection.

### 5.3 Orchestrator responsibilities

`code/benchmark.py:414` loops over rounds and backends. Framework modules are imported when dispatched. Each framework's `run_all_benchmarks` constructs duration/trial combinations, runs them, saves results incrementally, and prints a summary.

The orchestrator collects result dictionaries in memory, while CSV persistence protects completed measurements from being lost if a later case fails.

### 5.4 Backend keys

CLI dispatch keys and saved spike prefixes are not always the same:

| CLI flag | Dispatcher key | Saved backend/spike prefix |
|---|---|---|
| `--brian2-cpu` | `cpu` | `brian2cpp` |
| `--brian2cuda-gpu` | `gpu` | `brian2cuda` |
| `--pytorch` | `pytorch` | `pytorch` |
| `--nestgpu` | `nestgpu` | `nestgpu` |
| `--genn` | `genn` | `genn` |
| `--brian2genn` | `brian2genn` | `brian2genn` |

This distinction matters when locating files or specifying comparison backends.

## 6. Backend implementations

### 6.1 Brian2 CPU and Brian2CUDA

Both share `code/run_brian2_cuda.py` and its canonical equation/parameter definition.

**Standalone path:** CPU with one trial, or GPU with any trial count.

1. Reinitialize the Brian device.
2. Select C++ or CUDA standalone mode.
3. Load data and construct neurons, recurrent synapses, and Poisson inputs.
4. Schedule the simulation.
5. Generate and compile code.
6. Run the generated executable for each trial.
7. Extract spikes and write a combined Parquet.

The standalone implementation builds once per benchmark case and runs trials sequentially. See `code/run_brian2_cuda.py:193`.

**Parallel CPU path:** CPU with more than one trial.

The runner uses joblib's `loky` backend with `max(1, os.cpu_count() - 4)` workers. Every trial constructs its network in a worker process and runs Brian2 runtime simulation. Thus the path is not merely a larger batch of the one-trial C++ executable.

See `code/run_brian2_cuda.py:64` and `code/run_brian2_cuda.py:303`.

### 6.2 PyTorch

`code/run_pytorch.py` implements the network through custom `torch.nn.Module` classes:

- `PoissonSpikeGenerator`: external stimulation events.
- `AlphaSynapse`: decaying synaptic state and a delay buffer.
- `LIFNeuron`: voltage update, threshold, and reset.
- `AlphaLIF`: combines synaptic/neuron state with refractory counters.
- `TorchModel`: applies Poisson drive and sparse recurrent connectivity.

State tensors carry a trial/batch dimension. The weight matrix is shared, while state varies between trials. The runner chooses CUDA if `torch.cuda.is_available()` is true; otherwise it falls back to CPU.

A one-second simulation at the explicit 0.1 ms timestep requires 10,000 iterations. The outer simulation loop remains in Python, using tensor operations for each timestep.

The `ATan` autograd function supplies a surrogate derivative, but benchmark execution is wrapped in `torch.no_grad()` at `code/run_pytorch.py:420`. There is no implemented optimizer, training loss, or weight-learning procedure. Autograd-capable components do not by themselves make this a training system.

### 6.3 NEST GPU

`code/run_nestgpu.py` is both an importable orchestrator-facing runner and a subprocess worker.

Each trial launches the same file with `--worker`. Workers import NEST GPU, load the connectome, create `user_m1` neurons, connect recurrent edges, add Poisson inputs, simulate, and return JSON timing metadata. When recording is enabled, they also write temporary per-trial spike Parquets.

Trials are launched sequentially, not as simultaneous workers. The parent merges the successful per-trial Parquets and removes the temporary files.

The custom neuron implementation is in:

- `scripts/nestgpu_source_files/src/user_m1.h`.
- `scripts/nestgpu_source_files/src/user_m1.cu`.

The CUDA kernel updates voltage and synaptic drive, checks the threshold, resets state, and applies the refractory/silencing toggles. Its two input ports are arranged around `g_m` and `v_m`, allowing recurrent and external inputs to target different state variables.

The included Python binding passes connection weight arrays directly through the modified branch at `scripts/nestgpu_source_files/pythonlib/nestgpu.py:2225`.

Long recorded simulations are divided into 10,000 ms recording chunks, with a configured per-neuron recording capacity of 4,000 events per chunk. This is a practical buffer-management strategy, not proof that overflow is impossible under every future high-rate configuration.

NEST GPU failures trigger up to three whole-case attempts. Exhaustion records an error and exits the suite. See `code/run_nestgpu.py:369` and `code/run_nestgpu.py:521`.

### 6.4 Direct GeNN

`code/run_genn.py` uses PyGeNN to generate a custom CUDA model.

Its source contains:

- A Bernoulli external-source neuron model.
- A custom voltage-stimulated LIF neuron model.
- Sparse recurrent synapses with axonal delay.
- Sparse source-to-neuron activation targeting `Vstim`.
- `model.batch_size = n_run` for trial batching.

Each benchmark case constructs, builds, and loads a model. Recording buffers are periodically pulled from the device and accumulated into a final spike file.

The recording-window helper limits timestep-by-batch slots and chooses a window that divides the run length. This reduces the risk of excessive device recording memory for long, highly batched runs; it does not impose a fixed bound on all host-side accumulated spike data.

GeNN exposes both wall-clock simulation timing and an additional aggregate kernel-time measurement. The shared CSV uses the wall-clock simulation total, not the extra kernel measurement.

### 6.5 Brian2GeNN

`code/run_brian2_genn.py` reuses the Brian2 parameter definition but targets Brian2GeNN's standalone device.

To accommodate this backend, stimulation weights are stored as neuron variables and whole-population `PoissonInput` objects apply zero weight to non-selected neurons.

The runner rebuilds and runs separately for every trial. It sums mapping, network construction, build, and simulation timings across those trials. This is different from direct GeNN batching and Brian2CUDA's build-once loop.

It inserts deterministic C RNG seeds based on a base seed, trial number, and optional round number. It derives simulation time from the device's `_last_run_time`; if that value is unavailable, it falls back to the entire build/run wall time. That fallback changes the interpretation of the setup-versus-simulation split.

## 7. Trials, rounds, and benchmark grids

### 7.1 Default invocation

The default duration list is `[0.1, 1, 10, 100, 1000]` seconds. The default trial-count list is `[1, 4, 8, 16, 32]`.

This produces 25 cases per backend, or 125 cases across the five default backends in one round. Each case itself may contain multiple trials.

Running `python main.py` is therefore a broad benchmark suite, not a small smoke test.

### 7.2 Paper mode

`--paper` changes the default duration list to `[0.1, 1.0, 10.0, 100.0]` and the default round count to five. The trial-count list remains `[1, 4, 8, 16, 32]`.

- 20 cases per backend per round.
- 500 cases across five main backends and five rounds.
- 600 cases when all six backends' five-round outputs are combined.

Explicit `--t_run`, `--n_run`, or `--rounds` values can override paper defaults.

### 7.3 Labels and resume behavior

A label groups outputs under `data/results/<label>/`. Labeled runs use round subdirectories such as `round_01`.

If multiple rounds are requested without a label, the orchestrator generates a timestamp-based label. Labels are sanitized for filesystem use.

`--round-start` chooses the first round number written. It is useful for organizing resumed runs, but it is not an automatic checkpoint/recovery system: the runner does not scan completed cases and selectively skip them.

### 7.4 Randomness

A trial is intended to represent an independent realization, but identical stochastic input across frameworks is not guaranteed.

Direct GeNN uses `GENN_SEED`, defaulting to `12345`, without adding the round number in this runner. Repeated rounds may therefore reuse its configured seed for matching cases. Brian2GeNN explicitly varies its inserted C seed by trial and round. Other runners do not expose a unified seed policy here.

Separate timing rounds should not be automatically described as five independently seeded neural samples for every backend.

## 8. Output files and persistence rules

### 8.1 Spike output contract

New combined spike files are designed around:

| Column | Meaning |
|---|---|
| `t` | Legacy timing column; units differ by backend. |
| `time_ms` | Canonical new timing column in milliseconds. |
| `trial` | Zero-based trial identifier. |
| `neuron_index` | Internal neuron index. |
| `flywire_id` | FlyWire identity. |
| `exp_name` | Backend/duration/trial-count experiment filename stem. |

Each row represents one spike. Files are saved with Brotli Parquet compression.

The legacy `t` field is in seconds for Brian2, Brian2CUDA, NEST GPU, and Brian2GeNN exports, but in milliseconds for PyTorch and direct GeNN exports. New analysis should use `time_ms` or explicitly normalize units.

### 8.2 Typical paths

Examples of the naming contract:

```text
data/results/brian2cpp_t0.1s_n1.parquet
data/results/<label>/round_01/pytorch_t1.0s_n4.parquet
data/results/<label>/manifest.csv
```

The duration's string representation is part of the filename. The helpers do not canonicalize `1` and `1.0` to the same text. Consistency between run and comparison arguments matters.

### 8.3 Timing CSV

`data/benchmark-results.csv` contains the shared fields defined at `code/benchmark.py:150`:

- Framework, backend key, duration, and trial count.
- Setup, build, simulation, and total times.
- Realtime ratio.
- Recorded spikes and active-neuron count.
- Status and timestamp.
- Experiment identity, run label, round, and spike path.

The writer rounds timing fields and rewrites the CSV with the accumulated rows after each case. It is not append-only storage.

### 8.4 Replacement keys

Unlabeled rows use `(framework, n_run, t_run)` as the replacement key. Labeled/round-scoped rows also include experiment key, run label, and round.

Consequences:

- An unlabeled P9 run can replace an unlabeled sugar timing row with matching framework/duration/trial values.
- Spike filename stems do not include the biological experiment key, so two experiments sharing a label, round, backend, duration, and trial count can overwrite the same file.
- Timing-only mode is not a dedicated field in the shared key/schema. Use a distinct run label for timing-only measurements.

Use separate labels for scientifically distinct experiments and recording modes.

### 8.5 Manifests and hashes

A labeled result with a nonempty spike path triggers manifest writing. The manifest records the schema version and identifies `time_ms` as milliseconds.

The committed benchmark bundle also contains `checksums.sha256`. The current orchestrator does not generate that checksum file; the bundle includes it as separately retained metadata.

A manifest status or stored hash is evidence about an artifact's expected identity, not evidence that an absent local artifact has been verified.

### 8.6 Timing-only outputs

`--disable-spike-io` sets `FLY_BRAIN_DISABLE_SPIKE_IO=1`. The shared helper also treats `true`, `yes`, and `on` as disabling values.

This suppresses spike monitoring/recording and Parquet output, rather than merely moving file writes outside the timer. Spike and active-neuron counters become zero because they were not measured; zero in this mode does not mean the biological network was silent.

## 9. Timing methodology and interpretation

### 9.1 Main metrics

| Metric | Intended interpretation |
|---|---|
| `setup_time` | Data loading and network/model setup as reported by the backend. |
| `build_time` | Generated code compilation or equivalent device build. |
| `sim_time` | Backend-defined timed execution interval. |
| `total_time` | Backend-defined aggregate or wall time. |
| `realtime_ratio` | Aggregate simulated trial-seconds divided by `sim_time`. |

The ratio is an aggregate-throughput measure. For batched or parallel trials, it is not necessarily the latency ratio of a single continuously simulated animal.

### 9.2 Important timing asymmetries

- **Brian2 standalone:** compilation is separate; `sim_time` sums executable invocations; spike extraction is tracked outside those intervals.
- **Brian2 parallel CPU:** joblib wall time includes workers loading data, constructing networks, running simulation, and returning spikes. Setup/build fields are set to zero for this path.
- **PyTorch:** the timed loop includes spike detection and transfers of recorded spike indices to CPU when recording is enabled. Final Parquet writing is outside the loop.
- **NEST GPU:** simulation timing sums `Simulate` calls. Shared setup is the mean setup across trials, while simulation time is a sum. The recorded `total_elapsed` is captured before the parent merges combined Parquets.
- **Direct GeNN:** periodic recording-buffer transfers are separately accumulated as result collection; the timed simulation includes host stepping and synchronization.
- **Brian2GeNN:** separate builds are performed for each trial; its simulation/build split depends on availability of the device's private timing attribute.

Thus “file writing is outside `sim_time`” does not imply “all spike-related overhead is outside `sim_time`,” and similarly named timing columns are not identical experimental boundaries.

### 9.3 How to compare responsibly

Match experiment, data version, duration, trial count, recording mode, and environment. Distinguish one-trial latency from aggregate throughput. Report build-inclusive and simulation-only measurements separately.

A benchmark across these runners compares both frameworks and execution strategies. It is not a perfectly controlled experiment where only the framework name changes.

The timing-only dataset is valuable because it removes substantial observation/output work, but its result must not be mixed silently with normal recorded runs.

## 10. Accuracy and spike-output comparisons

### 10.1 Brian2 reference

The comparison layer treats `brian2cpp` as the reference implementation. That is computational ground truth for backend agreement, not direct experimental biological truth.

The README's upstream 91% prediction-accuracy statement is not a fresh measurement of this checkout, these six implementations, or the current P9 experiment.

### 10.2 Ground-truth comparison script

`code/compare_ground_truth.py` compares one duration/trial-count combination against Brian2 CPU. It calculates:

- Active-neuron Jaccard overlap, precision, and recall.
- Shared-active-neuron firing-rate Pearson correlation.
- Spike-count ratio.
- Relative firing-rate differences and the largest deviations.

Rates are total per-neuron counts divided by `t_run × n_run`. The script assigns heuristic `MATCH`, `CLOSE`, or `DEVIATED` labels using correlation and Jaccard thresholds.

Those labels are convenience classifications, not formal biological validation. The script also hardcodes a sugar-experiment description in its output metadata; that description should not be trusted unchanged for other experiment outputs.

### 10.3 Pairwise comparison script

`code/compare_spike_outputs.py` compares every available framework pair for selected grid points. It normalizes spike time, reuses per-file rate/time metrics, and produces:

- `pairwise_summary.csv`.
- `pairwise_summary.json`.
- `parity_rates.csv`.
- `missing_inputs.json`.

Timing matching is greedy and one-to-one within a tolerance, defaulting to 1 ms. It matches within the same trial and FlyWire neuron and reports match precision, recall, F1, and absolute timing differences.

### 10.4 Backend-versus-Brian2 across rounds

`code/compare_backend_to_brian2.py` generalizes the reference comparison across labeled rounds. Its default path reads only FlyWire IDs to calculate rate/count metrics, avoiding unnecessary timing work on large spike files.

`--include-timing` enables the more expensive spike-time matching path. Outputs are backend-prefixed summary, parity, and missing-input files.

### 10.5 Interpretation limitations

Shared-neuron correlation omits neurons active in only one output; inspect overlap and parity data too. Strong correlation can coexist with systematically different rates, so inspect count ratios and absolute errors.

Exact spike-time agreement is difficult to interpret when random input realizations are not synchronized. Low timing match can reflect differing stimulation sequences as well as differing dynamics.

None of these scripts constitutes an automated pass/fail test suite for all neuron-model edge cases.

## 11. Original-paper code and notebooks

The actual retained folder is `code/paper-phil-drosophila/`, not the `code/paper-brian2/` name shown in one README tree.

### 11.1 Original Brian2 model

`model.py` implements network creation, Poisson input, outgoing-connection silencing, per-trial simulation, joblib parallel experiments, and spike DataFrame export.

Its default trial count and activation rate differ from the benchmark defaults. Benchmark runners do not import this file; they contain their own implementations derived from the same model.

### 11.2 Analysis utilities

`utils.py` contains:

- `load_exps`: combine saved experiment Parquets.
- `get_rate`: calculate per-neuron mean firing rate and standard deviation across trials, including zero counts for trials in which a neuron did not spike.

These utilities use the legacy `t` contract rather than being the shared cross-backend normalization layer.

### 11.3 Example notebook

`example.ipynb` explores P9 activation, sugar activation, sugar/bitter coactivation, looming-related vision, JO-CE mechanosensory activation, and aversive olfaction. It names descending neurons whose activity could be relevant to body control.

Its content suggests a sensorimotor investigation direction, but there is no implemented conversion of those rates into body actuators.

### 11.4 Figures notebook

`figures.ipynb` contains figure-oriented activation and silencing sweeps using archived version-630 data, including sugar, water, bitter, and Johnston's-organ-related experiments.

Some data paths are written relative to a working directory that does not obviously match the notebook's current folder. Running notebooks may require working-directory/path adjustments. Their cells were inspected, not executed or reproduced during this review.

Tracked `.ipynb_checkpoints/` files are duplicate notebook snapshots, not additional backends.

## 12. Environments and setup dependencies

### 12.1 Main environment

`environment.yml` defines `brain-fly` with Python 3.10 and NumPy 1.26.4. It includes pandas, PyArrow, joblib, SciPy, Matplotlib, and notebook tools, plus pip-installed Brian2CUDA, PyTorch, and PyGeNN from the GeNN 5.4.0 source archive.

Several packages are not pinned to exact versions. The environment definition is not a complete reproducibility lockfile. The PyTorch pip source includes a CUDA-wheel extra index; that is an installation configuration, not proof of the CUDA capability of every resulting environment.

### 12.2 Brian2GeNN environment

`environment-brian2genn.yml` defines `brain-fly-brian2genn` with Python 3.10, NumPy 1.26.4, Brian2GeNN 1.7.0, data/analysis dependencies, and the GeNN 5.4.0 PyGeNN source archive.

The README and runner explain that Brian2GeNN additionally needs a GeNN 4.x command-line source tree, such as GeNN 4.9.0. Installing the PyGeNN 5.4 archive is not a substitute for that separate toolchain.

Brian2GeNN also imports the shared parameter module, which imports joblib. The separate environment does not explicitly list joblib; whether it arrives transitively should be checked rather than assumed.

### 12.3 NEST GPU installation

NEST GPU is not fully supplied by the main Conda file. The documented path is a source build that incorporates this repository's custom neuron and patched Python binding.

The runner searches configurable and conventional locations for the binding and native library. Installing a stock NEST GPU build without `user_m1` is not sufficient for this runner.

### 12.4 Environment controls

| Variable | Role |
|---|---|
| `FLY_BRAIN_DISABLE_SPIKE_IO` | Disable spike recording and export. |
| `CUDA_PATH`, `CUDA_HOME` | CUDA-toolkit discovery for generated builds. |
| `GENN_RECORDING_WINDOW_STEPS` | Requested GeNN recording window, default `100000`. |
| `GENN_RECORDING_WINDOW_MAX_SLOTS` | GeNN timestep/batch-slot limit, default `800000`. |
| `GENN_SEED` | Direct GeNN seed, default `12345`. |
| `BRIAN2GENN_GENN_PATH`, `GENN_PATH` | GeNN source-tree discovery for Brian2GeNN. |
| `BRIAN2GENN_SEED` | Brian2GeNN base C RNG seed, default `12345`. |
| `BRIAN2GENN_KEEP_BUILD` | Preserve Brian2GeNN build directories when set to `1`. |
| `NESTGPU_PYTHONLIB` | NEST GPU Python-binding location. |
| `NESTGPU_LIB` | NEST GPU native-library location. |

### 12.5 Setup script caveat

`scripts/setup_WSL_CUDA.sh` is a historical setup recipe mixing Windows WSL commands and Linux package/build commands. It also references an old environment name and an external `fly-body` checkout that is not part of this repository.

It should not be described as a verified, single-shell, end-to-end installation script for the current project.

### 12.6 Documented command examples

Run from the repository root after installing the appropriate environment. These examples reflect the CLI/source contract; only `--help` was executed during this review.

```bash
# Inspect options without importing simulation backends
python main.py --help

# Small, labeled CPU case without file logging
python main.py --brian2-cpu --t_run 0.1 --n_run 1 --no_log_file --run-label cpu_smoke

# P9 experiment using PyTorch
python main.py --pytorch --experiment p9 --t_run 1 --n_run 1 --no_log_file --run-label p9_check

# Timing-only PyTorch case under a distinct label
python main.py --pytorch --t_run 1 --n_run 4 --disable-spike-io --no_log_file --run-label sugar_noio_check

# Five-round main-backend grid; potentially expensive
python main.py --paper --run-label sugar_paper_review

# Separate-environment Brian2GeNN addition using the matching label
python main.py --brian2genn --paper --run-label sugar_paper_review

# Compare a labeled round after obtaining its spike files
python code/compare_spike_outputs.py --run-label sugar_paper_review --round 1
```

## 13. Current recorded results

### 13.1 Current top-level timing CSV

`data/benchmark-results.csv` currently contains two measurement rows:

| Framework | Duration | Trials | Status | Recorded observation |
|---|---:|---:|---|---|
| Brian2 CPU | 0.1 s | 1 | `success` | 1,696 spikes; 327 active neurons; 4.387 s simulation time; 15.616 s total time. |
| GeNN GPU | 0.1 s | 1 | `error: 'cuda'` | Setup time is recorded, but there is no successful simulation measurement or spike path. |

The Brian2 spike path exists locally. Its Parquet contents were not decoded in this session, so its count is attributed to the CSV, not independently recomputed.

The GeNN error was not reproduced or diagnosed. Its literal message does not establish whether the root cause is installation, backend discovery, or another environment problem.

### 13.2 Retained labeled spike-bundle metadata

`data/results/nature_2026_07/manifest.csv` contains:

- 600 unique expected grid entries.
- Six frameworks, with 100 entries per framework.
- Five rounds, with 120 entries per round.
- All entries marked `success`.
- All experiment keys set to `sugar`.
- No missing, duplicate, or unexpected framework/round/duration/trial-count combinations in the checked grid.

`checksums.sha256` contains 600 unique target paths. None of those 600 expected spike files was present locally during review.

The bundle README describes it as publication material. That description is repository-provided provenance; this review did not verify publication status, remote hosting, or the contents of the missing files.

### 13.3 Timing-only dataset

`data/results/nature_2026_07/no_io/timings.csv` contains 120 rows marked `success`, covering six frameworks with 20 rows each, one round, and the sugar experiment. It has no spike paths by design.

The stored aggregate summary reports:

| Framework | Median simulation reduction (%) | Median total reduction (%) |
|---|---:|---:|
| Brian2 CPU | 4.143 | 7.128 |
| Brian2CUDA | 11.237 | 13.469 |
| PyTorch CUDA | 9.572 | 7.904 |
| NEST GPU | −12.326 | 4.383 |
| GeNN GPU | 9.335 | 14.219 |
| Brian2GeNN GPU | 37.368 | 2.406 |

These are values read from `data/results/nature_2026_07/no_io/framework_summary.csv`, not newly measured or independently recomputed normal-versus-no-I/O statistics. Positive reduction means the no-I/O condition was faster under the summary's convention.

The NEST GPU row illustrates why recording removal should not be assumed to improve every simulation-only timing: its stored median simulation reduction is negative, despite a positive total reduction.

## 14. Implementation differences and operational risks

This section records source-level observations and interpretation risks. It is not a claim that every issue was reproduced as a failing runtime test.

### 14.1 Numerical equivalence is not automatic

Brian2 uses linear integration, while the custom tensor/CUDA models contain explicit step updates. NEST's threshold comparison is `>=`, while Brian2, PyTorch, and direct GeNN use strict `>` conditions in the inspected code.

Update ordering, floating-point precision, synaptic-delay quantization, and refractory behavior can affect spike times and propagation even when nominal parameters match.

In PyTorch, `AlphaLIF.forward` gates delayed input using a refractory condition, but passes voltage evolution through `LIFNeuron` without an equivalent refractory mask. Its equivalence to Brian2's `unless refractory` behavior needs focused tests; the class name alone is not proof.

### 14.2 Experiment semantics differ

Second-group and silencing differences are summarized in Section 4. A consistent experiment contract should define whether silencing means suppressing output spikes, removing outgoing connections, or removing all incident connections.

### 14.3 Active-neuron totals are not uniform

Brian2 and PyTorch combined outputs count unique active identities across all trials. NEST GPU reports the maximum per-trial active-neuron count at `code/run_nestgpu.py:451`, not the union across trials.

These can differ even when the merged files cover the same total set of neurons. Cross-framework comparisons should recompute a common definition from spike data.

### 14.4 Output identity and filename collisions

The current filename does not contain experiment identity or recording mode. Labels must be chosen deliberately. Numeric duration spelling can also lead to missing-file lookups between `1` and `1.0`.

### 14.5 Concurrent runs are unsafe by default

CSV and manifest writers use read-modify-rewrite operations without locks. Brian2 standalone removes the shared `code/output` tree before building. GeNN builds also reuse deterministic case directories.

Concurrent benchmark invocations in the same checkout can overwrite files, race on metadata, or delete another process's build artifacts. Separate working directories and isolated output paths would be safer.

### 14.6 Logging initialization

`BenchmarkLogger` opens the configured log file before the benchmark writer creates result directories. A fresh checkout with a missing custom log parent can fail at startup. Use console-only logging or ensure the parent directory exists.

### 14.7 Validation and failure signaling

The main CLI checks allowed durations and positive round numbers, but does not enforce positive `n_run` values. Some invalid values or failed cases are reported without a uniform nonzero process exit policy.

Several runners catch case-level exceptions and preserve error results while continuing. NEST GPU halts after exhausted retries. Completion messages must therefore be interpreted alongside per-row status, not as proof that every case succeeded.

### 14.8 Recording limits and memory

Chunking controls device recording buffers in GeNN and NEST GPU, but long runs can still accumulate large host-side results. PyTorch collects spike-index chunks during the loop. File output size and host memory remain relevant even when device buffers are bounded.

NEST's heuristic adjustment of chunk spike times and its recording ceiling deserve explicit chunk-boundary/overflow tests before extending to very different rates or durations.

### 14.9 Documentation drift

Examples of current drift include:

- The README tree's `paper-brian2` folder name differs from the actual `paper-phil-drosophila` folder.
- The README does not fully expose the current `--experiment` and timing-only options.
- The setup script contains outdated environment/path assumptions.
- The README's general silencing description is broader than several implementations' actual behavior.
- Frameworks described as ready still depend on particular native installations; the latest top-level CSV includes a GeNN failure.

Readiness should distinguish source implementation, historical run evidence, and a verified working environment on a specific machine.

## 15. Implemented versus not implemented

### 15.1 Implemented in source

- Current-connectome loading and identity mapping.
- Canonical Brian2 spiking-network construction.
- Six framework/backend execution paths.
- Two built-in benchmark experiments.
- Trial batching or parallel/sequential trial strategies.
- Repeated benchmark rounds and labels.
- Incremental timing persistence and labeled spike manifests.
- Spike export and long-run recording strategies.
- Timing-only execution.
- Reference and pairwise output-comparison scripts.
- Original-paper experiment and analysis notebooks.

### 15.2 Not implemented as a complete feature here

- A web UI, desktop experiment editor, or REST API.
- A packaged Python application/library with a tracked project manifest.
- General CLI/config-file selection of arbitrary neurons and activation rates.
- Uniform two-group stimulation and silencing across all backends.
- A unified deterministic cross-framework input schedule.
- Automatic case-level resume/skip logic.
- Concurrency-safe shared result storage.
- A committed automated test suite or CI workflow.
- A training loop, optimizer, synaptic learning rule, or adaptive connectome.
- A fly-body physics model integrated with brain outputs.
- Muscle actuation, body movement, or closed-loop sensory feedback.
- Independent reproduction of upstream biological prediction accuracy during this review.

These are scope boundaries, not all necessarily required defects. A pure simulation benchmark does not need to become a body-control application to be useful.

## 16. Suggested development priorities

The following are recommendations, not work performed or promises about future implementation.

### Priority 1: establish reproducible small runs

Verify one short labeled case for each backend in its intended environment. Preserve environment/package versions, hardware identity, recording mode, and complete failure diagnostics. Diagnose the recorded GeNN error before calling the current local GPU setup ready.

### Priority 2: formalize the shared model contract

Define timestep, integration/update order, threshold rule, reset, refractory behavior, stimulation, delay, and silencing in one explicit specification. Separate intended differences from accidental ones.

### Priority 3: add focused tests

Use small synthetic networks for tests—not invented benchmark results—to check:

- One-neuron leak, threshold, reset, and refractory behavior.
- Excitatory and inhibitory propagation.
- Exact delay scheduling.
- First/second stimulation groups.
- Uniform silencing semantics.
- Zero-spike export behavior.
- Time-unit normalization and filename identity.
- Multi-trial active-neuron aggregation.
- Recording chunk boundaries and overflow detection.
- CSV replacement keys and result-label isolation.

Whole-connectome runs can then serve as integration tests after small-model behavior is understood.

### Priority 4: improve provenance and measurement consistency

Store recording mode, effective device, timestep, data/cache fingerprints, seed policy, package versions, and hardware information with every result. Define consistent boundaries for setup, build, simulation, extraction, serialization, and total wall time.

### Priority 5: make experiment/output identity robust

Use a configurable experiment definition and ensure experiment key and recording mode participate in paths and metadata identity. Canonicalize duration formatting. Add atomic writes/locks or isolated outputs where simultaneous execution is needed.

### Optional direction: connect brain activity to a body

If the goal is an embodied fly, a separate integration layer would need to map selected descending-neuron activity to a controller, controller output to actuators, and body/environment observations back to sensory input. This is a new subsystem, not simply another existing benchmark flag.

## 17. Repository navigation guide

### 17.1 Main source inventory

Line counts below were computed from the reviewed files; they are size indicators, not coverage metrics.

| File | Lines | Primary role |
|---|---:|---|
| `main.py` | 202 | CLI entry point. |
| `code/benchmark.py` | 548 | Shared configuration, dispatch, logging, persistence. |
| `code/run_brian2_cuda.py` | 581 | Brian2 CPU and Brian2CUDA runners. |
| `code/run_pytorch.py` | 648 | Tensor model, sparse weights, batched runner. |
| `code/run_nestgpu.py` | 641 | NEST GPU worker lifecycle and result aggregation. |
| `code/run_genn.py` | 660 | Direct PyGeNN CUDA models and batching. |
| `code/run_brian2_genn.py` | 503 | Brian2GeNN independent build/run trials. |
| `code/compare_ground_truth.py` | 280 | Single-grid-point reference comparison. |
| `code/compare_spike_outputs.py` | 326 | All-framework pairwise comparison. |
| `code/compare_backend_to_brian2.py` | 341 | Across-round backend/reference parity. |
| `code/paper-phil-drosophila/model.py` | 372 | Retained original Brian2 experiment model. |
| `code/paper-phil-drosophila/utils.py` | 84 | Original experiment loading/rate analysis. |
| `scripts/nestgpu_source_files/pythonlib/nestgpu.py` | 3651 | Retained native-library binding with a connection-array patch. |

### 17.2 Useful source entry points

| Question | Start here |
|---|---|
| What experiment runs by default? | `code/benchmark.py:57` and `code/benchmark.py:101`. |
| What flags exist? | `main.py:52`. |
| Which backends run without flags? | `main.py:107`. |
| Where are model constants/equations? | `code/run_brian2_cuda.py:39`. |
| How are frameworks dispatched? | `code/benchmark.py:414`. |
| How are results keyed and saved? | `code/benchmark.py:207` and `code/benchmark.py:292`. |
| Where are spike paths generated? | `code/benchmark.py:180`. |
| How does PyTorch propagate activity? | `code/run_pytorch.py:247`. |
| How does the NEST CUDA neuron update? | `scripts/nestgpu_source_files/src/user_m1.cu:68`. |
| How does GeNN construct the network? | `code/run_genn.py:213`. |
| How does Brian2GeNN seed/run trials? | `code/run_brian2_genn.py:215`. |
| How can arbitrary Brian2 experiments be run? | `code/paper-phil-drosophila/model.py:295`. |
| Where is the retained benchmark evidence? | `data/results/nature_2026_07/manifest.csv` and `data/results/nature_2026_07/no_io/timings.csv`. |

### 17.3 Local source links

- [Root README](../README.md).
- [CLI entry point](../main.py).
- [Benchmark orchestrator](../code/benchmark.py).
- [Brian2/Brian2CUDA runner](../code/run_brian2_cuda.py).
- [PyTorch runner](../code/run_pytorch.py).
- [NEST GPU runner](../code/run_nestgpu.py).
- [Direct GeNN runner](../code/run_genn.py).
- [Brian2GeNN runner](../code/run_brian2_genn.py).
- [Ground-truth comparison](../code/compare_ground_truth.py).
- [Pairwise comparison](../code/compare_spike_outputs.py).
- [Across-round reference comparison](../code/compare_backend_to_brian2.py).
- [Original model](../code/paper-phil-drosophila/model.py).
- [Original utilities](../code/paper-phil-drosophila/utils.py).
- [Example notebook](../code/paper-phil-drosophila/example.ipynb).
- [Figures notebook](../code/paper-phil-drosophila/figures.ipynb).
- [Main environment](../environment.yml).
- [Brian2GeNN environment](../environment-brian2genn.yml).
- [Retained bundle manifest](../data/results/nature_2026_07/manifest.csv).
- [Timing-only raw measurements](../data/results/nature_2026_07/no_io/timings.csv).
- [Timing-only stored summary](../data/results/nature_2026_07/no_io/framework_summary.csv).

## 18. Verification performed

| Check | Outcome |
|---|---|
| Source branch and commit | Confirmed `Erfan` at the commit stated above. |
| Working-tree changes before authoring | No changes reported. |
| Stashed work | No stashes reported. |
| Repository-local agent instructions | No tracked `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` found in the inventory. |
| Python syntax | All 13 tracked Python files parsed successfully with `ast.parse`. |
| CLI help | `python main.py --help` completed successfully. |
| Current neuron CSV | 138,639 unique IDs; all rows marked complete. |
| Retained bundle grid | 600 unique expected combinations; no missing, duplicate, or unexpected points. |
| Checksum target inventory | 600 unique target paths; none of those targets locally present. |
| Timing-only raw dataset | 120 successful rows, six frameworks, one round, sugar experiment. |
| Automated tests / CI inventory | No tracked test suite or CI workflow found. |
| Simulation dependency availability | Required numerical/simulation packages absent from the review interpreter. |
| Full simulation execution | Not performed. |
| Missing bundle integrity verification | Not possible without its Parquet files. |
| Notebook/figure reproduction | Not performed. |

Syntax validation does not check imported dependency availability, generated C++/CUDA compilation, runtime semantics, or scientific correctness. A saved `success` row is historical evidence, not a new execution result from this review.

The large native-library binding was inspected at relevant integration points and syntax-checked; this was not a function-by-function audit of the entire third-party binding.

## 19. Licensing and provenance

The root `LICENSE` contains the GNU GPL version 2 text. The README states the overall project is GPL-2.0-or-later except where otherwise noted.

The retained original-paper directory has an MIT license naming Philip Shiu and Nico Spiller. Adapted NEST GPU files retain their GPL-2.0-or-later notices.

Preserve these separate notices when redistributing or reusing components. This document does not establish dataset licensing terms or provide a legal compatibility assessment.

The scientific references and remote bundle link in the root README are useful follow-up pointers. They were not externally retrieved or independently validated as part of this repository-understanding task.

## 20. Final assessment

This repository contains substantive implementations, not just scaffolding: a connectome-driven spiking model, multiple CPU/GPU execution strategies, neural-output export, repeated measurement infrastructure, and analysis tools.

Its strongest organizing idea is to evaluate **performance and output fidelity together**. A backend is useful not merely because it runs quickly, but because its simulated activity remains appropriately comparable to the reference under a well-defined experiment.

The main caution is that a shared dataset and nominal parameter table are not yet a complete equivalence guarantee. Experiment semantics, numerical updates, random stimulation, active-neuron counts, and timing boundaries differ in ways that should be made explicit and tested.

The current evidence supports describing this as a **whole-brain neural simulation and benchmarking project**. It does not support describing the checkout as a complete embodied fly, a trained AI system, a newly reproduced biological validation, or a freshly verified six-backend installation on the review machine.
