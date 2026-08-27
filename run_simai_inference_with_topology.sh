#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
用法:
  ./results/run_simai_inference_with_topology.sh \
      --workload <推理负载文件> \
      --topology <网络拓扑文件> \
      [--config <网络配置文件>] \
      [--output-dir <结果目录>] \
      [--threads <线程数>]

必选参数:
  -w, --workload FILE    推理负载文件，例如 results/Prefill/*.txt
  -n, --topology FILE    网络拓扑文件，例如 results/Prefill/Spectrum-X_*
      --network-topology FILE

可选参数:
  -c, --config FILE      SimAI 网络配置文件
      --network-config FILE
                         默认: 网络拓扑文件所在目录下的 SimAI.conf
  -o, --output-dir DIR   结果保存目录
                         默认: 网络拓扑文件所在目录/<负载文件名>/
  -t, --threads N        模拟器线程数，默认: 1
  -h, --help             显示帮助

示例:
  ./results/run_simai_inference_with_topology.sh \
      -w results/Prefill/Qwen3-Moe-235B-world_size32-tp4-pp2-ep2-bs1-seq512-prefill.txt \
      -n results/Prefill/Spectrum-X_32g_2gps_100Gbps_A100

  ./results/run_simai_inference_with_topology.sh \
      --workload /path/to/workload.txt \
      --topology /path/to/topology \
      --config /path/to/SimAI.conf \
      --output-dir /path/to/output
EOF
}

die() {
    echo "错误: $*" >&2
    exit 2
}

require_option_value() {
    local option=$1
    local value=${2:-}
    [[ -n "$value" ]] || die "参数 $option 缺少值"
}

workload_arg=""
topology_arg=""
config_arg=""
output_arg=""
threads=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--workload)
            require_option_value "$1" "${2:-}"
            workload_arg=$2
            shift 2
            ;;
        -n|--topology|--network-topology)
            require_option_value "$1" "${2:-}"
            topology_arg=$2
            shift 2
            ;;
        -c|--config|--network-config)
            require_option_value "$1" "${2:-}"
            config_arg=$2
            shift 2
            ;;
        -o|--output-dir)
            require_option_value "$1" "${2:-}"
            output_arg=$2
            shift 2
            ;;
        -t|--threads)
            require_option_value "$1" "${2:-}"
            threads=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            [[ $# -eq 0 ]] || die "不支持位置参数: $*"
            ;;
        *)
            die "未知参数: $1（使用 --help 查看用法）"
            ;;
    esac
done

[[ -n "$workload_arg" ]] || die "必须通过 --workload 指定推理负载文件"
[[ -n "$topology_arg" ]] || die "必须通过 --topology 指定网络拓扑文件"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || die "线程数必须是正整数: $threads"
[[ -f "$workload_arg" ]] || die "推理负载文件不存在: $workload_arg"
[[ -f "$topology_arg" ]] || die "网络拓扑文件不存在: $topology_arg"

# 在切换工作目录前将输入转换为绝对路径，支持从任意目录调用脚本。
workload_file=$(realpath --canonicalize-existing -- "$workload_arg")
topology_file=$(realpath --canonicalize-existing -- "$topology_arg")

if [[ -z "$config_arg" ]]; then
    config_arg="$(dirname -- "$topology_file")/SimAI.conf"
fi
[[ -f "$config_arg" ]] || die "网络配置文件不存在: $config_arg"
config_file=$(realpath --canonicalize-existing -- "$config_arg")

results_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$results_dir/.." && pwd)
simulator="$project_dir/bin/SimAI_simulator"
trace_reader_dir="$project_dir/ns-3-alibabacloud/analysis"
trace_reader="$trace_reader_dir/trace_reader"

[[ -x "$simulator" ]] || die "模拟器不存在或不可执行: $simulator"
[[ -x "$trace_reader" ]] || die "trace_reader 不存在或不可执行: $trace_reader"

workload_name=$(basename -- "$workload_file")
workload_stem=${workload_name%.*}
topology_name=$(basename -- "$topology_file")
safe_workload_stem=$(printf '%s' "$workload_stem" | sed 's/[^A-Za-z0-9._-]/_/g')

if [[ -z "$output_arg" ]]; then
    output_arg="$(dirname -- "$topology_file")/$safe_workload_stem"
fi

mkdir -p -- "$output_arg"
run_dir=$(cd -- "$output_arg" && pwd)
parameter_dir="$run_dir/parameters"
mkdir -p -- "$parameter_dir"

trace_output_value=$(
    awk '$1 == "TRACE_OUTPUT_FILE" { value = $2 } END { print value }' "$config_file"
)
[[ -n "$trace_output_value" ]] || die "网络配置中缺少 TRACE_OUTPUT_FILE: $config_file"

if [[ "$trace_output_value" == /* ]]; then
    generated_trace="$trace_output_value"
else
    generated_trace="$project_dir/$trace_output_value"
fi

echo "[1/4] 保存本次推理参数"
cp -f -- "$workload_file" "$parameter_dir/workload-${workload_name}"
cp -f -- "$topology_file" "$parameter_dir/topology-${topology_name}"
cp -f -- "$config_file" "$parameter_dir/SimAI.conf"
rm -f -- /etc/astra-sim/SimAI.log

echo "[2/4] 运行 SimAI 推理模拟"
echo "  Workload: $workload_file"
echo "  Topology: $topology_file"
echo "  Config:   $config_file"
cd -- "$project_dir"
AS_SEND_LAT="${AS_SEND_LAT:-3}" \
AS_NVLS_ENABLE="${AS_NVLS_ENABLE:-1}" \
AS_LOG_LEVEL="${AS_LOG_LEVEL:-DEBUG}" \
    "$simulator" \
    -t "$threads" \
    -w "$workload_file" \
    -n "$topology_file" \
    -c "$config_file"

[[ -f "$generated_trace" ]] || die "模拟结束后未生成二进制 trace: $generated_trace"
[[ -f "$project_dir/ncclFlowModel_EndToEnd.csv" ]] || \
    die "模拟结束后未生成 ncclFlowModel_EndToEnd.csv"
[[ -f "$project_dir/ncclFlowModel_test1_dimension_utilization_0.csv" ]] || \
    die "模拟结束后未生成维度利用率 CSV"

echo "[3/4] 收集模拟结果到 $run_dir"
cp -f -- "$project_dir/ncclFlowModel_EndToEnd.csv" "$run_dir/"
cp -f -- "$project_dir/ncclFlowModel_test1_dimension_utilization_0.csv" "$run_dir/"

if [[ -d /etc/astra-sim ]]; then
    cp -r -- /etc/astra-sim "$run_dir/"
fi

# 配置允许把 trace 写到 /etc/astra-sim 之外，因此单独保存一份固定名称。
saved_binary_trace="$run_dir/trace.tr"
cp -f -- "$generated_trace" "$saved_binary_trace"

echo "[4/4] 解析二进制 trace"
text_trace="$run_dir/trace.txt"
"$trace_reader" "$saved_binary_trace" > "$text_trace"

echo
echo "推理模拟完成"
echo "  Workload: $workload_file"
echo "  Topology: $topology_file"
echo "  Config:   $config_file"
echo "  参数目录: $parameter_dir"
echo "  结果目录: $run_dir"
echo "  文本 trace: $text_trace"
