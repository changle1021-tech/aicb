SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

model_size=deepseek-671B
config_file_path=
phase=decode
seq_length=1024
micro_batch=32
world_size=8
tensor_model_parallel_size=1
expert_model_parallel_size=
pipeline_model_parallel=1
moe_enable=true
result_dir=results/workload/
aiob_enable=false
aiob_forward_loops=10

dpsk_default_path="$SCRIPT_DIR/inference_configs/deepseek_default.json"
dpsk_v2_lite_default_path="$SCRIPT_DIR/inference_configs/deepseek_v2_lite_chat.json"
qwen3_moe_default_path="$SCRIPT_DIR/inference_configs/qwen3_moe_default.json"
qwen3_next_default_path="$SCRIPT_DIR/inference_configs/qwen3_next_default.json"
qwen3_32b_default_path="$SCRIPT_DIR/inference_configs/qwen3_32b.json"

usage() {
  cat <<-EOF
Usage: $0 [OPTIONS]

  Generate inference workload with AIOB support.

Options:
  -m, --model-size <SIZE>
      Model size to use.
      Possible values: {deepseek-671B, deepseek-v2-lite-chat, qwen3-235B,
                        qwen3-next-80B, qwen3-32B}.
      (Default: $model_size)

  -c, --config <FILE>
      Path to a custom configuration file.
      (Default: None)

  -p, --phase <PHASE>
      Inference phase.
      Possible values: {prefill, decode}.
      (Default: $phase)

  -s, --seq-length <LENGTH>
      Sequence length for the model.
      (Default: $seq_length)

  -b, --micro-batch <SIZE>
      Micro batch size.
      (Default: $micro_batch)

  -w, --world-size <SIZE>
      Total number of GPUs (world size).
      (Default: $world_size)

  -t, --tp-size <SIZE>
      Tensor model parallel size.
      (Default: $tensor_model_parallel_size)

  -e, --ep-size <SIZE>
      Expert model parallel size (for MoE models).
      (Default: 8 for MoE models; 1 for dense Qwen3-32B)

  -l, --pp-size <SIZE>
      Pipeline model parallel size.
      (Default: $pipeline_model_parallel)

  -M, --moe-enable
      Enable MoE (Mixture of Experts) support.
      (This is a boolean flag. Default: $moe_enable)

  -r, --result-dir <DIR>
      Directory to save the results.
      (Default: "$result_dir")

  -a, --aiob-enable
      Enable AIOB (All-In-One Block) support.
      (This is a boolean flag. Default: $aiob_enable)

  -f, --aiob-loops <LOOPS>
      Number of forward loops for AIOB.
      (Default: $aiob_forward_loops)

  -h, --help
      Display this help message and exit.

Example:
  sh $0 -m deepseek-671B -p decode -s 1024 -b 32 --aiob-enable

EOF
  exit "${1:-1}"
}

while [ $# -gt 0 ]
do
  case $1 in
    -m|--model_size|--model-size)
      model_size=$2; shift;;
    -c|--config)
      config_file_path=$2; shift;;
    -p|--phase)
      phase=$2; shift;;
    -s|--seq_length|--seq-length)
      seq_length=$2; shift;;
    -b|--micro_batch|--micro-batch)
      micro_batch=$2; shift;;
    -w|--world_size|--world-size)
      world_size=$2; shift;;
    -t|--tensor_model_parallel_size|--tp-size)
      tensor_model_parallel_size=$2; shift;;
    -e|--expert_model_parallel_size|--ep-size)
      expert_model_parallel_size=$2; shift;;
    -l|--pipeline_model_parallel|--pp-size)
      pipeline_model_parallel=$2; shift;;
    -M|--moe_enable|--moe-enable)
      moe_enable=true;;
    -r|--result_dir|--result-dir)
      result_dir=$2; shift;;
    -a|--aiob_enable|--aiob-enable)
      aiob_enable=true;;
    -f|--aiob_forward_loops|--aiob-loops)
      aiob_forward_loops=$2; shift;;
    -h|--help)
      usage 0;;
    (*)
      echo "Unknown option: $1" >&2
      usage;;
  esac
  shift
done

case $model_size in
  deepseek-671B)
    model_name=DeepSeek-671B
    config_file_path=${config_file_path:-$dpsk_default_path}
    expert_model_parallel_size=${expert_model_parallel_size:-8}
    ;;
  deepseek-v2-lite-chat)
    model_name=DeepSeek-V2-Lite-Chat
    config_file_path=${config_file_path:-$dpsk_v2_lite_default_path}
    expert_model_parallel_size=${expert_model_parallel_size:-8}
    ;;
  qwen3-235B)
    model_name=Qwen3-Moe-235B
    config_file_path=${config_file_path:-$qwen3_moe_default_path}
    expert_model_parallel_size=${expert_model_parallel_size:-8}
    ;;
  qwen3-next-80B)
    model_name=Qwen3-Next-80B
    config_file_path=${config_file_path:-$qwen3_next_default_path}
    expert_model_parallel_size=${expert_model_parallel_size:-8}
    ;;
  qwen3-32B|qwen-32B|qwen32B)
    model_name=Qwen3-32B
    config_file_path=${config_file_path:-$qwen3_32b_default_path}
    expert_model_parallel_size=${expert_model_parallel_size:-1}
    if [ "$expert_model_parallel_size" != "1" ]; then
      echo "Qwen3-32B is a dense model; --expert_model_parallel_size must be 1." >&2
      exit 2
    fi
    moe_enable=false
    ;;
  (*)
    echo "Invalid model size: $model_size"
    usage;;
esac

# Build command with optional parameters. Positional arguments preserve spaces
# in paths and avoid evaluating shell metacharacters.
if command -v python3 >/dev/null 2>&1; then
  python_bin=python3
else
  python_bin=python
fi

if [ "$model_name" = "Qwen3-32B" ]; then
  set -- "$python_bin" "$SCRIPT_DIR/qwen3_dense_workload_generator.py" "$model_name" "$config_file_path"
else
  set -- "$python_bin" -m workload_generator.SimAI_inference_workload_generator "$model_name" "$config_file_path"
fi

# Add optional parameters if they are set
if [ ! -z "$phase" ]; then
  set -- "$@" --phase "$phase"
fi

if [ ! -z "$seq_length" ]; then
  set -- "$@" --seq_length "$seq_length"
fi

if [ ! -z "$micro_batch" ]; then
  set -- "$@" --micro_batch "$micro_batch"
fi

if [ ! -z "$world_size" ]; then
  set -- "$@" --world_size "$world_size"
fi

if [ ! -z "$tensor_model_parallel_size" ]; then
  set -- "$@" --tensor_model_parallel_size "$tensor_model_parallel_size"
fi

if [ ! -z "$expert_model_parallel_size" ] && [ "$model_name" != "Qwen3-32B" ]; then
  set -- "$@" --expert_model_parallel_size "$expert_model_parallel_size"
fi

if [ ! -z "$pipeline_model_parallel" ]; then
  set -- "$@" --pipeline_model_parallel "$pipeline_model_parallel"
fi

if [ "$moe_enable" = true ]; then
  set -- "$@" --moe_enable
fi

if [ ! -z "$result_dir" ]; then
  set -- "$@" --result_dir "$result_dir"
fi

if [ "$aiob_enable" = true ]; then
  set -- "$@" --aiob_enable
fi

if [ ! -z "$aiob_forward_loops" ]; then
  set -- "$@" --aiob_forward_loops "$aiob_forward_loops"
fi

printf 'Running:'
printf ' %s' "$@"
printf '\n'
"$@"

# A one-rank expert-parallel group cannot produce inter-rank traffic. Keep the
# MoE compute stages, but normalize their EP collectives to no-ops.
if [ "$expert_model_parallel_size" = "1" ] && [ "$moe_enable" = true ]; then
  workload_file="$result_dir/${model_name}-world_size${world_size}-tp${tensor_model_parallel_size}-pp${pipeline_model_parallel}-ep${expert_model_parallel_size}-bs${micro_batch}-seq${seq_length}-${phase}.txt"
  if [ -f "$workload_file" ]; then
    "$python_bin" "$SCRIPT_DIR/normalize_ep1_workload.py" "$workload_file"
  fi
fi
