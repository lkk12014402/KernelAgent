#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional


class RunResult:
    def __init__(
        self,
        run_id: str,
        is_success: bool,
        reason: str,
        num_subgraphs: Optional[int],
    ):
        self.run_id = run_id
        self.is_success = is_success
        self.reason = reason
        self.num_subgraphs = num_subgraphs  # None 表示没拿到（比如 subgraphs.json 无效）


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON {path}: {e}")
        return None


def check_single_run(run_dir: Path) -> RunResult:
    """
    返回 RunResult，其中：
      - is_success: 该 run 是否成功
      - reason: 文本说明
      - num_subgraphs: 子图数量（若 subgraphs.json 无效，则为 None）
    判定逻辑：
      - subgraphs.json 必须存在且为 array；
      - 若 len(subgraphs) == 1: 用 kernels_out/summary.json 单条 success 判定；
      - 若 len(subgraphs) != 1: 若有 compose_out/summary.json，用其 success 判定；
        否则失败。
    """
    run_id = run_dir.name
    subgraphs_path = run_dir / "subgraphs.json"
    kernels_summary_path = run_dir / "kernels_out" / "summary.json"
    compose_summary_path = run_dir / "compose_out" / "summary.json"

    # 1. 检查 subgraphs.json
    if not subgraphs_path.is_file():
        reason = f"{run_id}: missing subgraphs.json"
        return RunResult(run_id, False, reason, None)

    subgraphs = load_json(subgraphs_path)
    if not isinstance(subgraphs, list):
        reason = f"{run_id}: subgraphs.json is not a JSON array"
        return RunResult(run_id, False, reason, None)

    n_subgraphs = len(subgraphs)

    # 情况 A：只有一个子图 -> 用 kernels_out/summary.json 判定
    if n_subgraphs == 1:
        if not kernels_summary_path.is_file():
            reason = f"{run_id}: single subgraph but missing kernels_out/summary.json"
            return RunResult(run_id, False, reason, n_subgraphs)

        ks = load_json(kernels_summary_path)
        if not isinstance(ks, list):
            reason = f"{run_id}: kernels_out/summary.json is not a JSON array"
            return RunResult(run_id, False, reason, n_subgraphs)

        if len(ks) != 1:
            reason = f"{run_id}: expected 1 kernel summary entry, got {len(ks)}"
            return RunResult(run_id, False, reason, n_subgraphs)

        record = ks[0]
        success = bool(record.get("success", False))
        kernel_path_str = record.get("kernel_path")
        if not success:
            reason = f"{run_id}: kernel summary success=False"
            return RunResult(run_id, False, reason, n_subgraphs)

        if not kernel_path_str:
            reason = f"{run_id}: kernel_path missing in kernel summary"
            return RunResult(run_id, False, reason, n_subgraphs)

        kernel_path = Path(kernel_path_str)
        if not kernel_path.is_file():
            reason = f"{run_id}: kernel_path does not exist: {kernel_path}"
            return RunResult(run_id, False, reason, n_subgraphs)

        # 满足所有条件，认为这个 run 成功
        reason = f"{run_id}: single-subgraph kernel OK"
        return RunResult(run_id, True, reason, n_subgraphs)

    # 情况 B：多个子图 -> 看 compose_out/summary.json
    else:
        if not compose_summary_path.is_file():
            reason = f"{run_id}: multi-subgraph but missing compose_out/summary.json"
            return RunResult(run_id, False, reason, n_subgraphs)

        cs = load_json(compose_summary_path)
        # 假设 compose_out/summary.json 是一个 dict，含有 success 字段
        if not isinstance(cs, dict):
            reason = f"{run_id}: compose_out/summary.json is not a JSON object"
            return RunResult(run_id, False, reason, n_subgraphs)

        success = bool(cs.get("success", False))
        if not success:
            reason = f"{run_id}: compose summary success=False"
            return RunResult(run_id, False, reason, n_subgraphs)

        reason = f"{run_id}: multi-subgraph composition OK"
        return RunResult(run_id, True, reason, n_subgraphs)


def scan_all_runs(base_dir: Path) -> Dict[str, RunResult]:
    """
    遍历 base_dir 下的所有 run 目录（比如 .fuse/run_xxx），
    返回 {run_id: RunResult}。
    """
    results: Dict[str, RunResult] = {}
    if not base_dir.is_dir():
        print(f"[ERROR] Base directory not found: {base_dir}", file=sys.stderr)
        return results

    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        result = check_single_run(run_dir)
        results[run_id] = result
        print(f"[INFO] {result.reason}")

    return results


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print(f"Usage: {sys.argv[0]} <base_run_dir>", file=sys.stderr)
        print("Example: python check_accuracy.py .fuse", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(argv[0]).resolve()
    results = scan_all_runs(base_dir)

    if not results:
        print("[WARN] No run directories found. Accuracy is undefined.")
        return

    total = len(results)
    success_count = sum(1 for r in results.values() if r.is_success)
    accuracy = success_count / total if total > 0 else 0.0

    # 统计子图数量分布
    single_subgraph_runs = sum(
        1 for r in results.values() if r.num_subgraphs == 1
    )
    multi_subgraph_runs = sum(
        1 for r in results.values() if r.num_subgraphs is not None and r.num_subgraphs > 1
    )
    unknown_subgraph_runs = sum(
        1 for r in results.values() if r.num_subgraphs is None
    )

    print("\n=== Summary ===")
    print(f"Base dir      : {base_dir}")
    print(f"Total runs    : {total}")
    print(f"Success runs  : {success_count}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Runs with 1 subgraph   : {single_subgraph_runs}")
    print(f"Runs with >1 subgraphs : {multi_subgraph_runs}")
    print(f"Runs with unknown #subgraphs (bad subgraphs.json): {unknown_subgraph_runs}")

    print("\nPer-run status:")
    for run_id in sorted(results.keys()):
        r = results[run_id]
        status = "SUCCESS" if r.is_success else "FAIL"
        ns = "unknown" if r.num_subgraphs is None else str(r.num_subgraphs)
        print(f"{run_id}: {status}, num_subgraphs={ns}")


if __name__ == "__main__":
    main()
