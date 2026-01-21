#!/usr/bin/env bash

set -ex
set -euo pipefail

# ===== 配置区：按需修改 =====
PYTHON_BIN="${PYTHON_BIN:-python}"   # 或者 /home/xxx/anaconda3/envs/xxx/bin/python
KERNELAGENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELBENCH_ROOT="${KERNELAGENT_ROOT}/../KernelBench/KernelBench"

EXTRACT_MODEL="DeepSeek-R1-G2-static-671B"
DISPATCH_MODEL="DeepSeek-R1-G2-static-671B"
COMPOSE_MODEL="DeepSeek-R1-G2-static-671B"
WORKERS=4
MAX_ITERS=5
LLM_TIMEOUT_S=3000
RUN_TIMEOUT_S=600
#TARGET_PLATFORM="cuda"
TARGET_PLATFORM="xpu"
DISPATCH_JOBS="auto"
# ============================

cd "${KERNELAGENT_ROOT}"

problems=()
for level in level1 level2 level3; do
  lvl_dir="${KERNELBENCH_ROOT}/${level}"
  if [[ -d "${lvl_dir}" ]]; then
    while IFS= read -r -d '' f; do
      problems+=("$f")
    done < <(find "${lvl_dir}" -maxdepth 1 -type f -name '*.py' -print0 | sort -z)
  fi
done

total=${#problems[@]}
echo "Found ${total} problem files under ${KERNELBENCH_ROOT}."

i=0
for problem in "${problems[@]}"; do
  i=$(( i + 1 ))
  echo "==============================="
  echo "[${i}/${total}] Problem: ${problem}"
  echo "==============================="

  cmd=(
    "${PYTHON_BIN}"
    -m Fuser.pipeline
    --problem "$(realpath "${problem}")"
    --extract-model "${EXTRACT_MODEL}"
    --dispatch-model "${DISPATCH_MODEL}"
    --dispatch-jobs "${DISPATCH_JOBS}"
    --compose-model "${COMPOSE_MODEL}"
    --workers "${WORKERS}"
    --max-iters "${MAX_ITERS}"
    --llm-timeout-s "${LLM_TIMEOUT_S}"
    --run-timeout-s "${RUN_TIMEOUT_S}"
    --target-platform "${TARGET_PLATFORM}"
    --verify
  )

  echo "Running: ${cmd[*]}"
  # 直接把 pipeline 的 stdout/stderr 打到当前终端
  if ! "${cmd[@]}"; then
    echo "[FAIL] pipeline failed for ${problem}" >&2
  else
    echo "[OK] pipeline finished for ${problem}"
  fi
done

echo "All done."
