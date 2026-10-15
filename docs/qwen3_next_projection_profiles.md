# Qwen3-Next projection profiling

The projection generator derives local widths from model configuration instead
of using individual head dimensions as complete projection sizes. For the
default configuration at TP1:

| Projection | K | N |
| --- | ---: | ---: |
| GDN QKVZ | 2048 | 12288 |
| GDN b/a | 2048 | 64 |
| GDN output | 4096 | 2048 |
| Full-attention QKV, output gate enabled | 2048 | 9216 |
| Full-attention output | 4096 | 2048 |

GDN output quantization uses the complete local value-head width. Full-attention
KV heads are replicated when TP exceeds the number of KV heads, as in vLLM.
Disabling `attn_output_gate` gives QKV N=5120 at TP1. Query normalization and the
attention kernel still use 16 query heads; the gate does not add attention heads.

## Precision and output gate

`workload_generator/Vidur_workload_generator.py` accepts the following
Qwen3-Next options:

- `--projection_precision bf16`: BF16 linear projections, without FP8 input
  quantization. This is the default for a `torch_dtype=bfloat16` model config.
- `--projection_precision fp8`: retain the explicit DeepGEMM FP8 projection path
  and its input quantization. This is a component-level profiling control,
  not a claim that an entire FP8 serving model uses these kernels.
- `--attn_output_gate` / `--no_attn_output_gate`: override the model config.
  An omitted option preserves the config, with vLLM's default of enabled.

The full-attention output sigmoid and multiplication are profiled together in
`atten_output_gate`. The existing statistics parser includes this entry once in
`attention_layer`; it is not also added to `atten_o`.

Projection/gate callbacks use CUDA Graph device events around the complete
callback, including every kernel that a linear implementation emits. Input
creation and JIT/warmup are outside the measured interval. A 128 MiB cache flush
precedes the start event on each replay; its time is excluded. The helper returns
the median of 30 measurements in **seconds**. Existing AIOB output converts this
to microseconds, the workload CSV uses nanoseconds, and Vidur reads seconds.
This isolated device region does not measure model-forward CPU launch gaps.

Other operators retain their existing profiling paths. In particular, choosing
BF16 projections does not change routed-expert DeepGEMM kernels or make the whole
profile equivalent to a BF16 serving runtime.

## Existing profiles

Generated AIOB statistics and workload CSVs receive a `.projections.json` sidecar
containing version, precision, local shapes, invocation parameters, timing scope,
source identity, and a SHA256 of the actual statistics/CSV payload.

The Qwen3-Next workload entry point rejects missing or incompatible statistics
metadata. Existing projection timings cannot be repaired by attaching new
metadata or changing a dtype label: regenerate them explicitly with the corrected
generator on permitted GPU resources. Missing tables do not automatically enable
profiling in this entry point.

Vidur's independent CSV/JSON/in-memory caches are outside this AICB change. They
must be regenerated/invalidated separately before expecting simulations to use
new timings. An already cached Vidur table will not be updated by checking out
this commit. Runtime/library/GPU provenance for the entire profile also needs
validation when producing replacement tables; this sidecar identifies only the
projection change, not every component in the workload.

## CPU verification

From the repository root, with no GPU dependencies required:

```bash
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tests -p 'test_qwen3_next_projection*.py' -v
```

The suite verifies TP1/2/4 widths, KV replication, invalid head layouts, precision
dispatch, query-gate math, actual AIOB forward argument wiring, configuration
overrides, gate aggregation and units, and rejection of legacy/mismatched
profiles. GPU callbacks are replaced with CPU fakes; heavy AIOB methods are
extracted from the production source to test their wiring without importing
CUDA packages.

These tests do not validate new GPU kernel execution, CUDA Graph capture on a
particular device, or post-fix TTFT/TPOT. Those require a subsequent run on
permitted resources. Convolution, GDN recurrent/chunk paths, shared experts,
communication, and worker overhead remain separate issues.
