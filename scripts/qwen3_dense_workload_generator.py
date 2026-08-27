"""Generate a SimAI inference workload for a dense Qwen3 model.

The generic inference generator currently models only MoE Qwen variants. A
dense Qwen3 block has no expert dispatch/combine traffic: tensor parallelism
introduces one all-reduce after the attention output projection and one after
the MLP down projection.
"""

from __future__ import annotations

import argparse
import json
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
    num_hidden_layers: int, tensor_parallel_size: int, communication_size: int
) -> list[WorkItem]:
    collective = "ALLREDUCE" if tensor_parallel_size > 1 else "NONE"
    collective_size = communication_size if tensor_parallel_size > 1 else 0
    items: list[WorkItem] = []
    for _ in range(num_hidden_layers):
        items.extend(
            (
                WorkItem("attention_norm"),
                WorkItem(
                    "attention_layer",
                    forward_comm=collective,
                    forward_comm_size=collective_size,
                ),
                WorkItem("mlp_norm"),
                WorkItem(
                    "dense_mlp",
                    forward_comm=collective,
                    forward_comm_size=collective_size,
                ),
            )
        )
    return items


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
    parser.add_argument("--pipeline_model_parallel", type=positive_int, default=1)
    parser.add_argument("--result_dir", type=Path, default=Path("results/workload"))
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

    activation_bytes = active_tokens * config["hidden_size"] * BYTES_PER_BFLOAT16
    items = generate_items(
        config["num_hidden_layers"],
        args.tensor_model_parallel_size,
        activation_bytes,
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
