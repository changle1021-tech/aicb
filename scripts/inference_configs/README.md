# Adding an Inference Model

Use this checklist when adding another inference model to AICB. The goal is to
model the compute and communication of the real serving stack, rather than to
make the simulated total match by applying a fixed correction.

## Integration points

A model normally touches the following files:

1. Add the architecture JSON to this directory.
2. Register its aliases, default config, and parallelism defaults in
   `scripts/inference_workload_with_aiob.sh`.
3. Add or select a workload generator that emits the SimAI operations.
4. Add an AIOB hardware profiler when measured compute time is required.
5. Add a usage example to the inference section of the top-level `README.md`.

The shell entry point currently has two generator paths. Existing generic
inference models use `workload_generator.SimAI_inference_workload_generator`;
dense Qwen3-32B uses the standalone
`scripts/qwen3_dense_workload_generator.py`. Register a new model explicitly in
the appropriate path, and do not silently route an unsupported architecture
through an approximately similar model.

## 1. Define the profiling unit

Document whether the profiler returns:

- one GPU/rank or the complete distributed job;
- one Transformer layer or the complete model;
- one request, one micro batch, or an averaged token;
- compute kernels only or end-to-end framework latency.

AIOB inference profiles should normally represent one rank and one layer. The
workload generator expands that result over the model layers and emits
communication separately. Do not multiply by the layer count in both places,
and do not divide a per-layer measurement by pipeline-parallel size unless the
workload format explicitly requires it.

## 2. Keep prefill and decode separate

Prefill and decode have different shapes and often use different kernels:

```text
prefill active tokens = micro_batch * sequence_length
decode active tokens  = micro_batch
decode KV length      = sequence_length
```

Prefill normally uses causal prefill attention. Decode normally reads a paged
KV cache and processes one new token per sequence. A decode-specific timing or
kernel change must be guarded by the phase so that an already calibrated
prefill path is unchanged.

Test both phases after every shared-profiler change. A decode improvement is
not acceptable if it creates a new prefill regression.

## 3. Match the production kernels and data types

Use the same execution path as the real inference runtime whenever possible:

- FlashAttention/SDPA for prefill only when that is the selected backend;
- FlashInfer, FlashAttention paged attention, or FlashMLA for decode according
  to the serving stack;
- the runtime's fused RMSNorm, RoPE, activation, routing, and grouped-GEMM
  operators;
- the actual BF16, FP16, FP8, INT8, or INT4 representation;
- the real dense, MoE, MLA, GQA, or linear-attention architecture.

Do not silently replace paged decode attention with dense PyTorch SDPA, or a
quantized GEMM with BF16 `linear`. A fallback is allowed only when it models the
same operation closely and the profile records which fallback was used.

Runtime APIs can move between releases. For example, a vLLM operation may be
available through `vllm._custom_ops` in one version and `torch.ops._C` in
another. Resolve supported locations deliberately and report a clear error
when a required fused kernel or attention backend is unavailable.

## 4. Validate TP, PP, DP, and EP geometry

Check at least the following constraints:

- hidden and intermediate dimensions are divisible where tensor parallelism
  shards them;
- local query-head and KV-head counts are correct;
- KV heads are replicated when TP exceeds the global KV-head count;
- query heads remain divisible by local KV heads for GQA;
- expert ownership, routing top-k, and expert parallel groups are correct;
- world size is consistent with the supported TP/PP/DP/EP decomposition;
- pipeline parallelism changes layer placement, not the shape of an unrelated
  single-layer kernel.

The hardware profiler is usually a single-process, single-GPU program. Passing
`--world_size 4 --tensor_model_parallel_size 4` selects the local TP4 tensor
shapes; it does not automatically launch four profiling processes. Distributed
communication belongs in the generated workload unless the profiler is
explicitly designed to measure it.

## 5. Keep the timed region precise

For steady-state kernel profiling:

- warm up JIT compilation, algorithm selection, and caches first;
- perform attention planning and workspace initialization outside the timed
  region;
- preallocate output and temporary tensors when production reuses them;
- exclude Python dispatch gaps and unrelated CUDA Event overhead;
- use Kineto/CUPTI kernel intervals, or another method with equivalent
  semantics, for very short decode kernels;
- include every CUDA kernel launched by a logical fused operation, but do not
  count the same child kernel twice;
- avoid `empty_cache()` between steady-state decode components;
- use enough loops to produce stable results.

Do not remove allocations or clones merely because the total is too large.
First determine whether the real serving path performs them and whether the
current timer is accidentally including them. Conversely, do not include input
preparation, JIT compilation, or attention planning when they do not occur per
token in production.

For in-place operators, prepare stable inputs outside the timed region. Reusing
mutated uninitialized tensors can produce NaNs or overflow even when only
latency is being measured.

Low utilization in `nvidia-smi` is not by itself a profiler failure. Batch-one
decode consists of small, often memory- or launch-limited kernels, and the
default utilization sampling window can miss microsecond GPU bursts. Validate
the recorded kernel times instead of increasing batch size only to make the GPU
look busy.

## 6. Model the KV cache without wasting memory

Allocate paged decode KV storage from the requested sequence length:

```text
pages_per_sequence = ceil(sequence_length / page_size)
total_pages         = micro_batch * pages_per_sequence
```

Build valid `indptr`, page indices, and last-page lengths for every sequence.
Use the layout expected by the selected backend. Do not allocate a fixed,
oversized page pool when a compact cache represents the same kernel geometry;
it can cause OOM failures and distort cache state or setup time.

Initialize backend workspaces according to their API contract. Workspace
creation and `plan()` normally stay outside the measurement.

## 7. Preserve units and workload mappings

The current dense Qwen3 path uses this contract:

```text
AIOB profile: microseconds
SimAI compute field: nanoseconds
conversion: microseconds * 1000, exactly once
```

Verify every profile section is mapped to the intended workload item. Norm,
attention, MLP, routing, and expert kernels must not be omitted or included in
two different sections. Communication sizes are bytes and should reflect the
local activation shape for the selected phase and parallel topology.

Use result filenames that include model, phase, sequence length, batch, TP, PP,
and EP as applicable. Different profiles must not overwrite one another unless
their measured shapes and semantics are identical.

Record enough metadata to reproduce the result:

```text
device
dtype or quantization
phase
profile loops
timing mode
attention backend
relevant library versions
```

## 8. Validation matrix

Before considering a model supported, cover at least:

- prefill and decode;
- TP1 and every intended TP configuration, such as TP2 and TP4;
- PP1 and at least one multi-stage PP case;
- short, medium, and long sequence lengths;
- batch one and the intended production batch where applicable;
- one run with newly collected AIOB data and one run that reloads the saved
  profile;
- missing-dependency and out-of-memory error paths.

For accuracy comparisons, hold these real-runtime settings constant:

- model weights and quantization;
- batch size and active request count;
- prompt/KV length;
- attention backend;
- CUDA Graph mode;
- TP/PP/EP topology;
- GPU type, clocks, and software versions.

Compare per-component times before comparing only the total. Investigate large
differences at the operator/backend level; never introduce a topology-specific
fixed millisecond offset as the primary correction.

## 9. Minimum checks before commit

Run checks proportional to the changed path:

```bash
python3 -m py_compile <profiler.py> <workload_generator.py>
sh scripts/inference_workload_with_aiob.sh --help
git diff --check
```

On the target GPU, collect fresh profiles for the validation matrix and inspect
both the AIOB output and generated SimAI workload. Confirm that decode-only
changes leave prefill results stable. Commit source changes separately from
generated profiles unless the generated artifacts are intentionally part of
the change.
