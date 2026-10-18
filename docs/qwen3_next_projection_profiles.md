# Qwen3-Next profiles and execution boundaries

The validated configuration is Qwen3-Next-80B-A3B BF16, TP1/PP1/DP4/EP4,
one active request, prefill chunks of at most 512 tokens, H100 and vLLM 0.11.2.
The original projection shape helpers also cover TP2/4; that does not imply that
the complete runtime calibration has been validated at those configurations.

## Profiling and units

At TP1 the GDN QKVZ projection is [2048,12288], b/a is [2048,64], and
output is [4096,2048]. Full-attention QKV (including output gate) is
[2048,9216], and output is [4096,2048]. The default projection precision
follows the model BF16 dtype. Explicit FP8 projection microbenchmarks are a
separate experiment; the full native GDN prefill chain is BF16.

Prefill uses eager stream regions, retaining host launch gaps. GDN profiling
includes compiled QKV ordering/materialization, the native GDN op, projections,
gated norm and both decoder norms. The CSV writer removes the separately
measured norms for GDN layers, so they are included once. Full attention uses
vLLM's paged FA3 path, K/V cache writes, Q/K norm and the output gate. The
hybrid BF16 cache page size is derived from the GDN state; it is 544 at TP1.

Decode uses CUDA Graph timing. Events are outside replay: compare a graph of
32 (128 MiB flush + callback) pairs with 32 flushes, alternating measurement
order. Divide the median difference by 32. Events captured inside each callback
inflate short kernels; warm repetitions without per-call flush can retain
weights in L2. Input allocation and warmup/compilation are outside the interval.

The helpers return seconds; AIOB statistics use microseconds; workload CSV
`comp_time` uses nanoseconds. Vidur converts CSV values back to seconds.

## MoE and communication

BF16 routed experts use native fused-expert kernels. Expert routing is over
DP-padded gathered tokens, while the router linear uses local tokens. Shared
experts run on local tokens and can overlap routed experts on another stream;
adding their separate timings is not the runtime critical path.

`--moe_compute_profile PATH` supplies measurements for all 48 layers from real
routes. The profile checks model layout, producer identity and (for combined
native profiles) runtime source identity. `compute_only` profiles receive
separate EP AG(hidden, router) + RS(hidden) communication in Vidur.
`compute_and_ep_comm` profiles already include it and receive zero additional
EP time. Combined timing must not be described as compute-only timing. Native FusedMoE
replay begins after the external router linear; the profiler retains that linear
when `router_linear_included_by_phase` is false. Compute-only calibration
already includes it.

The communication payload is BF16 hidden states plus router logits, not a
DeepEP FP8 top-k payload. A runtime profile is specific to four ranks padded to
the same size, even if only one rank owns a request. Independent collectives and
independent compute do not necessarily add to the complete MoE duration.

## Identity, cache and context

Statistics and CSVs carry `.projections.json` sidecars with version 3, precision,
shapes, phase/context, invocation, timing scope, four component-source hashes,
MoE calibration hash and the actual payload hash. A changed payload or source
requires regeneration. Do not repair old timings by changing their metadata.

Vidur's Qwen loader bypasses the old persistent/global lookup caches. It checks
all 48 attention/MoE rows and rejects missing data, incompatible layouts,
nonfinite values and unsupported batch/PP settings. Interpolation is linear
between measured points and independent of request history; out-of-range
queries fail. Prefill has separate chunk and existing-context axes. Decode's
sequence axis is the current KV length.

## CPU and request boundaries

CPU modeling is enabled with
`--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling`
and requires `--random_forrest_execution_time_predictor_config_aicb_cpu_overhead_profile PATH`.
The old five empirical CPU formulas are bypassed for Qwen. Inclusive CPU
bookkeeping can wait on the GPU and must not be added to model time again.

Worker perimeter v2 is host iteration wall time minus model GPU elapsed. It is
kept for reproducing earlier experiments. Request perimeter v3 distinguishes
first prefill, later prefill and decode. It adds measured client/worker request
IO once at context=0, and measured inter-chunk gaps at later chunks. Request IO
comes from aligned monotonic timestamps, not an E2E residual fit. The schema
and timing scope must agree; v1 GPU-only boundaries are rejected.

## Reproduction

From the AICB root, using the vLLM 0.11.2 environment and an independently
validated four-rank MoE profile:

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
python -m workload_generator.Vidur_workload_generator \
  Qwen3-Next-80B scripts/inference_configs/qwen3_next_default.json \
  --seq_length 512 --context_length 0 --micro_batch 1 --world_size 4 \
  --tensor_model_parallel_size 1 --expert_model_parallel_size 4 \
  --moe_compute_profile /absolute/path/moe_profile.json \
  --aiob_enable --phase prefill --result_dir /absolute/path/workload
```

CPU verification (GPU callbacks are replaced by fakes):

```bash
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tests -p 'test_qwen3_next*.py' -v
```

These tests establish contracts and wiring. GPU correctness, observer overhead
and prediction errors require the corresponding raw measurements and a
separate uninstrumented validation set; unit test success alone establishes
none of those results.
