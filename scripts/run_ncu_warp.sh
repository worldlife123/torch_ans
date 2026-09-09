#!/usr/bin/env bash
# A1 — profile the warp-level interleaved pop kernel with Nsight Compute.
#
#   ./scripts/run_ncu_warp.sh --strategy sparse --q 11
#   ./scripts/run_ncu_warp.sh --strategy binary --rows 4096 --out /tmp/a1
#
# Extra flags are forwarded to scripts/profile_warp_pop.py (profile mode), so
# every launch geometry / lookup strategy can be profiled with the same entry
# point. The report is written to <out>/ncu_warp_pop_<strategy>_q<q>.ncu-rep.
#
# GPU performance counters need CAP_SYS_ADMIN when the driver runs with
# RmProfilingAdminOnly=1 (the default on GeForce). The script detects that and
# prints the exact sudo command to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DRIVER="$ROOT/scripts/profile_warp_pop.py"

# the workspace build (CUDA + invcdf ops) must win over an older installed one
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

pick_ncu() {
  local c
  for c in \
    "${NCU:-}" \
    "${CONDA_PREFIX:-$HOME/miniconda3}/pkgs/nsight-compute-2024.3.2.3-0/nsight-compute-2024.3.2/ncu" \
    /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu \
    /usr/lib/nsight-compute/ncu \
    /usr/bin/ncu
  do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

NCU_BIN="$(pick_ncu)" || { echo "no ncu binary found (set NCU=/path/to/ncu)"; exit 2; }

# ---- parse the few flags we need, the rest goes to the python driver -------
STRATEGY="sparse"; Q="11"; OUT="${OUT:-$ROOT/.ncu}"
FWD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --strategy) STRATEGY="$2"; FWD+=("$1" "$2"); shift 2 ;;
    --q) Q="$2"; FWD+=("$1" "$2"); shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) FWD+=("$1"); shift ;;
  esac
done

LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-2}"
mkdir -p "$OUT"
REPORT="$OUT/ncu_warp_pop_${STRATEGY}_q${Q}"

# NOTE: do NOT use "--set full": it pulls in PmSampling / PmSampling_WarpStates,
# which makes ncu fall back to a single PM-sampling pass (# Pass Groups 1) and
# silently drop the classic warp-stall breakdown (smsp__average_warps_issue_
# stalled_*). Explicit sections keep the multi-pass kernel replay.
SECTIONS_DEFAULT=(
  SpeedOfLight ComputeWorkloadAnalysis MemoryWorkloadAnalysis
  MemoryWorkloadAnalysis_Chart MemoryWorkloadAnalysis_Tables SchedulerStats
  WarpStateStats InstructionStats LaunchStats Occupancy SourceCounters
  WorkloadDistribution
)
if [ -n "${NCU_SECTIONS:-}" ]; then
  # e.g. NCU_SECTIONS="PmSampling PmSampling_WarpStates" for the sampled warp
  # state distribution (the classic smsp__average_warps_issue_stalled_* counters
  # are not collectable on this driver/tool combination, see SPARSE_INVCDF_SUMMARY.md)
  IFS=' ' read -r -a SARG <<< "$NCU_SECTIONS"
  SECTION_ARGS=()
  for s in "${SARG[@]}"; do SECTION_ARGS+=(--section "$s"); done
elif [ -n "${NCU_SET:-}" ]; then
  SECTION_ARGS=(--set "$NCU_SET")
else
  SECTION_ARGS=()
  for s in "${SECTIONS_DEFAULT[@]}"; do SECTION_ARGS+=(--section "$s"); done
fi

IFS=' ' read -r -a EXTRA <<< "${NCU_EXTRA:-}"

CMD=("$NCU_BIN"
  --target-processes all
  "${SECTION_ARGS[@]}"
  --clock-control none
  -k regex:rans_warp_pop_indexed_kernel
  -c "$LAUNCH_COUNT"
  -o "$REPORT"
  --force-overwrite
  "${EXTRA[@]}"
  "$PY" "$DRIVER" profile "${FWD[@]}")

if [ "$(id -u)" != "0" ] && grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null; then
  echo "== GPU performance counters are admin-only (RmProfilingAdminOnly: 1). =="
  echo "== Re-run the command below as root (or set the module option to 0). =="
  echo
  echo "sudo env HOME=$HOME PATH=\"$PATH\" CONDA_PREFIX=\"${CONDA_PREFIX:-}\" PYTHONPATH=\"$PYTHONPATH\" \\"
  echo "  $(printf '%q ' "${CMD[@]}")"
  echo
  echo "persistent alternative (survives reboot, needs a reload):"
  echo "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf"
  exit 3
fi

echo "ncu: $NCU_BIN ($("$NCU_BIN" --version | grep -o 'Version [0-9.]*'))"
"${CMD[@]}"
echo "report: ${REPORT}.ncu-rep"
