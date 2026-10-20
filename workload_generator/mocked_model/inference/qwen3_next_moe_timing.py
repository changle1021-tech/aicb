"""Compose measured Qwen MoE work without serializing an overlapped branch.

Raw component measurements are preserved. This helper only removes the shared
branch when the measured routed compute alone already covers its duration.
Dispatch makes that routed branch longer, so communication may still be added
by the existing simulator without moving the critical branch in this case.
Uncertain runtime/precision/shape cases retain the serial estimate.
"""
import hashlib
import json
import math
from pathlib import Path
import re

REQUIRED = ('moe_route_gate', 'moe_route_select_experts',
            'moe_expert_routed', 'moe_shared_expert')


def read_moe_component_ns(profile_path, phase):
    values = {}
    current_phase = current_key = None
    for line in Path(profile_path).read_text().splitlines():
        match = re.match(r'^\s*inference_term\s*:\s*(\S+)\s*$', line)
        if match:
            current_phase = match.group(1)
            current_key = None
            continue
        match = re.match(r'^\s*(\w+)\s*:\s*$', line)
        if match:
            current_key = match.group(1)
            continue
        match = re.match(r'^\s*time_gpu_avg\s*:\s*(\S+)\s*$', line)
        if match and current_phase == phase and current_key in REQUIRED:
            value = float(match.group(1)) * 1000.0  # raw microseconds -> ns
            if not math.isfinite(value) or value < 0 or current_key in values:
                raise ValueError('Invalid/duplicate MoE timing: '+current_key)
            values[current_key] = value
            current_key = None
    return values


def compose_shared_expert(cache, components, *, local_tokens, threshold,
                          enabled, precision):
    result = dict(cache)
    details = dict(applied=False, local_tokens=local_tokens,
                   shared_stream_threshold=threshold, precision=precision,
                   component_ns=dict(components))
    if not enabled:
        details['reason'] = 'serial_requested_or_unverified_runtime'
    elif precision != 'bf16':
        details['reason'] = 'precision_path_not_verified'
    elif threshold is None or local_tokens > threshold:
        details['reason'] = 'shared_expert_uses_main_stream'
    elif any(key not in components for key in REQUIRED):
        details['reason'] = 'separate_component_measurements_missing'
    else:
        main = sum(components[key] for key in REQUIRED[:3])
        shared = components['moe_shared_expert']
        if main < shared:
            # Dispatch/combine timing is not available at workload generation.
            # Keep a conservative serial estimate instead of guessing its split.
            details['reason'] = 'shared_branch_requires_joint_communication_profile'
        else:
            result['moe_expert'] = round(components['moe_expert_routed'])
            details.update(applied=True, reason='routed_compute_covers_shared_branch',
                           routed_compute_ns=main, shared_compute_ns=shared)
    return result, details


def compose_qwen_moe_cache(cache, profile_path, args):
    """Gate the overlap rule on the actual runtime used by this profile."""
    from workload_generator.mocked_model.inference.qwen3_next_projections import projection_precision
    from importlib import metadata
    runtime = metadata.version('vllm')
    mode = getattr(args, 'moe_shared_overlap', 'auto')
    # Verified against this user's installed serving implementation. Unknown
    # implementations keep their old estimate and remain usable at any batch.
    enabled = mode == 'auto' and runtime.split('+')[0] == '0.11.2'
    threshold = None
    if enabled:
        from vllm import envs
        threshold = envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    phase = args.phase
    tokens = args.micro_batch if phase == 'decode' else args.seq_length
    components = read_moe_component_ns(profile_path, phase)
    result, details = compose_shared_expert(cache, components,
        local_tokens=tokens, threshold=threshold, enabled=enabled,
        precision=projection_precision(args))
    details.update(vllm=runtime, phase=phase, requested_mode=mode,
        assumptions=['non-modular BF16 FusedMoE', 'DP chunking disabled',
                     'separate shared-expert stream as in vLLM 0.11.2'])
    return result, details


def write_composition_metadata(csv_path, raw_profile_path, details):
    csv_path, raw_path = Path(csv_path), Path(raw_profile_path)
    writer = Path(__file__).parents[2] / 'Vidur_workload_generator.py'
    payload = dict(details, schema='qwen_moe_shared_overlap_v1',
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        raw_profile_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        transform_sha256=hashlib.sha256(Path(__file__).read_bytes()+writer.read_bytes()).hexdigest(),
        timing_scope='isolated device components; shared branch overlap composition',
        limits='routing statistics and communication/compute contention are not corrected')
    Path(str(csv_path)+'.moe_timing.json').write_text(json.dumps(payload,indent=2)+'\n')
