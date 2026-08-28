"""Generate a SimAI inference workload for a dense Qwen3 model.

The generic inference generator currently models only MoE Qwen variants. A
dense Qwen3 block has no expert dispatch/combine traffic: tensor parallelism
introduces one all-reduce after the attention output projection and one after
the MLP down projection.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BYTES_PER_BFLOAT16 = 2
DEFAULT_COMPUTE_TIME = 1


@dataclass(frozen=True)
class WorkItem:
    name: str
    forward_compute_time: int = DEFAULT_COMPUTE_TIME
    forward_comm: str = "NONE"
    forward_comm_size: int = 0

    def serialize(self) -> str:
        # SimAI workload v1.1 has twelve tab-separated fields.
        return "\t".join(
            map(
                str,
                (
                    self.name,
                    -1,
                    self.forward_compute_time,
                    self.forward_comm,
                    self.forward_comm_size,
                    0,
                    "NONE",
                    0,
                    0,
                    "NONE",
                    0,
                    100,
                ),
            )
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    required = {
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"config is missing required fields: {', '.join(missing)}")
    if config.get("model_type") != "qwen3":
        raise ValueError(
            f"expected a dense qwen3 config, got model_type={config.get('model_type')!r}"
        )
    if "num_experts" in config or "moe_intermediate_size" in config:
        raise ValueError("MoE configs must use the generic inference workload generator")
    return config


def validate_topology(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.expert_model_parallel_size != 1:
        raise ValueError(
            "dense Qwen3 requires expert_model_parallel_size to be 1"
        )
    model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel
    if args.world_size % model_parallel_size:
        raise ValueError(
            "world_size must be divisible by tensor_model_parallel_size * "
            "pipeline_model_parallel"
        )
    if config["hidden_size"] % args.tensor_model_parallel_size:
        raise ValueError("hidden_size must be divisible by tensor model parallel size")
    if config["num_attention_heads"] % args.tensor_model_parallel_size:
        raise ValueError(
            "num_attention_heads must be divisible by tensor model parallel size"
        )
    if args.seq_length > config["max_position_embeddings"]:
        raise ValueError(
            f"seq_length {args.seq_length} exceeds max_position_embeddings "
            f"{config['max_position_embeddings']}"
        )


def generate_items(
    num_hidden_layers: int,
    tensor_parallel_size: int,
    communication_size: int,
    compute_times: dict[str, int] | None = None,
) -> list[WorkItem]:
    compute_times = compute_times or {}
    collective = "ALLREDUCE" if tensor_parallel_size > 1 else "NONE"
    collective_size = communication_size if tensor_parallel_size > 1 else 0
    items: list[WorkItem] = []
    for _ in range(num_hidden_layers):
        items.extend(
            (
                WorkItem(
                    "attention_norm",
                    forward_compute_time=compute_times.get(
                        "attention_norm", DEFAULT_COMPUTE_TIME
                    ),
                ),
                WorkItem(
                    "attention_layer",
                    forward_compute_time=compute_times.get(
                        "attention_layer", DEFAULT_COMPUTE_TIME
                    ),
                    forward_comm=collective,
                    forward_comm_size=collective_size,
                ),
                WorkItem(
                    "mlp_norm",
                    forward_compute_time=compute_times.get(
                        "mlp_norm", DEFAULT_COMPUTE_TIME
                    ),
                ),
                WorkItem(
                    "dense_mlp",
                    forward_compute_time=compute_times.get(
                        "dense_mlp", DEFAULT_COMPUTE_TIME
                    ),
                    forward_comm=collective,
                    forward_comm_size=collective_size,
                ),
            )
        )
    return items


def get_profile_path(args: argparse.Namespace) -> Path:
    filename = (
        f"{args.model_name}-world_size{args.world_size}"
        f"-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}"
        f"-ep1-bpg{args.micro_batch}-seq{args.seq_length}-{args.phase}.txt"
    )
    return args.aiob_output_dir / filename


def load_compute_times(profile_path: Path) -> dict[str, int]:
    section_pattern = re.compile(r"^(\w+):\s*$")
    current_section: str | None = None
    averages: dict[str, float] = {}
    with profile_path.open(encoding="utf-8") as profile:
        for raw_line in profile:
            section_match = section_pattern.match(raw_line)
            if section_match:
                current_section = section_match.group(1)
                continue
            if current_section and raw_line.strip().startswith("time_gpu_avg:"):
                averages[current_section] = float(raw_line.split(":", 1)[1].strip())

    # Profile values are microseconds; SimAI compute fields are nanoseconds.
    attention_norm = averages.get("atten_norm", 0.0)
    attention = sum(
        value
        for name, value in averages.items()
        if name.startswith("atten_") and name != "atten_norm"
    )
    mlp_norm = averages.get("mlp_norm", 0.0)
    mlp = sum(
        value
        for name, value in averages.items()
        if name.startswith("mlp_") and name != "mlp_norm"
    )
    result = {
        "attention_norm": round(attention_norm * 1000),
        "attention_layer": round(attention * 1000),
        "mlp_norm": round(mlp_norm * 1000),
        "dense_mlp": round(mlp * 1000),
    }
    return {name: value for name, value in result.items() if value > 0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a SimAI inference workload for dense Qwen3 models"
    )
    parser.add_argument("model_name")
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--seq_length", type=positive_int, default=1024)
    parser.add_argument("--micro_batch", type=positive_int, default=1)
    parser.add_argument("--world_size", type=positive_int, default=1)
    parser.add_argument("--tensor_model_parallel_size", type=positive_int, default=1)
    parser.add_argument("--expert_model_parallel_size", type=positive_int, default=1)
    parser.add_argument("--pipeline_model_parallel", type=positive_int, default=1)
    parser.add_argument("--result_dir", type=Path, default=Path("results/workload"))
    parser.add_argument("--aiob_enable", action="store_true")
    parser.add_argument("--aiob_forward_loops", type=positive_int, default=10)
    parser.add_argument(
        "--aiob_output_dir", type=Path, default=Path("results/aiob_outputs")
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config_file)
        validate_topology(args, config)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    # Decode models one new token per sequence. Prefill processes the complete
    # prompt for every sequence in the micro batch.
    active_tokens = args.micro_batch
    if args.phase == "prefill":
        active_tokens *= args.seq_length

    profile_path = get_profile_path(args)
    if args.aiob_enable:
        try:
            from qwen3_dense_aiob import ProfilingError, profile_dense_qwen3

            profile_path = profile_dense_qwen3(
                config=config,
                model_name=args.model_name,
                phase=args.phase,
                seq_length=args.seq_length,
                micro_batch=args.micro_batch,
                world_size=args.world_size,
                tensor_parallel_size=args.tensor_model_parallel_size,
                pipeline_parallel_size=args.pipeline_model_parallel,
                loops=args.aiob_forward_loops,
                output_dir=args.aiob_output_dir,
            )
        except ProfilingError as error:
            parser.error(str(error))

    compute_times: dict[str, int] = {}
    if profile_path.is_file():
        try:
            compute_times = load_compute_times(profile_path)
        except (OSError, ValueError) as error:
            parser.error(f"could not read AIOB profile {profile_path}: {error}")
        print(f"using AIOB compute profile: {profile_path}")
    else:
        print("AIOB profile not found; using default compute time")

    activation_bytes = active_tokens * config["hidden_size"] * BYTES_PER_BFLOAT16
    items = generate_items(
        config["num_hidden_layers"],
        args.tensor_model_parallel_size,
        activation_bytes,
        compute_times,
    )
    pp_comm_size = activation_bytes if args.pipeline_model_parallel > 1 else 0

    args.result_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{args.model_name}-world_size{args.world_size}"
        f"-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}"
        f"-ep1-bs{args.micro_batch}-seq{args.seq_length}-{args.phase}.txt"
    )
    output_path = args.result_dir / filename
    header = (
        "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
        f"model_parallel_NPU_group: {args.tensor_model_parallel_size} "
        f"ep: 1 pp: {args.pipeline_model_parallel} all_gpus: {args.world_size} "
        "mode: 1 vpp: 1 ga: 1 checkpoints: 0 checkpoint_initiates: 0 "
        f"pp_comm: {pp_comm_size}"
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{header}\n{len(items)}\n")
        output.writelines(f"{item.serialize()}\n" for item in items)

    print(f"workload saved in: {output_path}")


if __name__ == "__main__":
    main()
