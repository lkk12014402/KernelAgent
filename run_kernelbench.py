#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 配置部分：根据你自己的环境调整
KERNELAGENT_ROOT = Path(__file__).resolve().parent
KERNELBENCH_ROOT = KERNELAGENT_ROOT.parent / "KernelBench" / "KernelBench"
ARTIFACTS_ROOT = KERNELAGENT_ROOT / "artifacts"

# 默认模型与参数（你可以按需要调整）
DEFAULT_EXTRACT_MODEL = "DeepSeek-R1-G2-static-671B"
DEFAULT_DISPATCH_MODEL = "DeepSeek-R1-G2-static-671B"
DEFAULT_COMPOSE_MODEL = "DeepSeek-R1-G2-static-671B"
DEFAULT_WORKERS = 4
DEFAULT_MAX_ITERS = 5
DEFAULT_LLM_TIMEOUT_S = 3000
DEFAULT_RUN_TIMEOUT_S = 600
DEFAULT_TARGET_PLATFORM = "cuda"
DEFAULT_DISPATCH_JOBS = "auto"

PYTHON_EXEC = sys.executable  # 使用当前环境的 Python


def _run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run a command and capture stdout/stderr."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def _infer_level_and_name(problem_path: Path) -> Tuple[str, str]:
    """
    从 ../KernelBench/KernelBench/levelX/YY_Name.py 推断 level 和 name。
    例如：
      level = "L1"
      name  = "19_ReLU"
    """
    # level1 / level2 / level3
    level_dir = problem_path.parent.name  # "level1"
    m = re.match(r"level(\d+)", level_dir)
    level = f"L{m.group(1)}" if m else level_dir

    stem = problem_path.stem  # "19_ReLU"
    return level, stem


def _find_latest_fuse_run() -> Optional[Path]:
    """
    在当前工作目录下的 .fuse 中找到按修改时间排序的最新 run 目录。
    假设你每次 pipeline 都写到同一个 .fuse 根下。
    """
    base = Path.cwd() / ".fuse"
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    # 按修改时间排序，取最新
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_artifacts_from_run(run_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    从单个 run_dir 中抽取：
      - composed_kernel.py 路径
      - composition_summary.json 内容
    """
    compose_out = run_dir / "compose_out"
    composed = compose_out / "composed_kernel.py"
    summary = compose_out / "composition_summary.json"

    if not composed.is_file():
        return None, None
    summary_data: Optional[Dict[str, Any]] = None
    if summary.is_file():
        try:
            summary_data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            summary_data = None
    return composed, summary_data


def _write_single_artifact_set(
    problem_path: Path,
    composed_kernel_path: Path,
    summary_data: Optional[Dict[str, Any]],
    out_root: Path,
) -> None:
    """将单个问题的结果写到 artifacts/<level>/<name>/ 目录。"""
    level, name = _infer_level_and_name(problem_path)
    out_dir = out_root / level / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. original.py：从 KernelBench 复制
    original_dst = out_dir / "original.py"
    shutil.copy2(problem_path, original_dst)

    # 2. kernel_code.py：从 composed_kernel.py 复制
    kernel_dst = out_dir / "kernel_code.py"
    shutil.copy2(composed_kernel_path, kernel_dst)

    # 3. manifest.json
    manifest: Dict[str, Any] = {
        "level": level,
        "name": name,
        "files": {
            "original": "original.py",
            "kernel": "kernel_code.py",
        },
        # pipeline 路径固定标记为 "pipeline"
        "route": "pipeline",
        "verify": {},
    }

    if summary_data:
        # 尽量贴一些关键信息进去
        verify_obj: Dict[str, Any] = {
            "success": bool(summary_data.get("success", False)),
            "rounds": summary_data.get("rounds"),
            "target_platform": summary_data.get("target_platform"),
            "details": {
                "composition": {
                    "verify_passed": summary_data.get("verify_passed"),
                    "verify_reason": summary_data.get("verify_reason"),
                    "validator": summary_data.get("validator"),
                    "stdout_path": summary_data.get("stdout_path"),
                    "stderr_path": summary_data.get("stderr_path"),
                }
            },
        }
        manifest["verify"] = verify_obj

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Saved artifacts for {level}/{name} -> {out_dir}")


def main():
    # 遍历 level1/2/3 下的所有 .py
    problem_files: list[Path] = []
    for lvl in ["level1", "level2", "level3"]:
        lvl_dir = KERNELBENCH_ROOT / lvl
        if not lvl_dir.is_dir():
            continue
        for p in sorted(lvl_dir.glob("*.py")):
            problem_files.append(p)

    if not problem_files:
        print(f"No problem files found under {KERNELBENCH_ROOT}", file=sys.stderr)
        return 1

    print(f"Found {len(problem_files)} problem files.")

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, problem_path in enumerate(problem_files, start=1):
        level, name = _infer_level_and_name(problem_path)
        print(f"\n=== [{idx}/{len(problem_files)}] {level}/{name} ===")
        # 已经存在就跳过的话，可以解开这里
        out_dir = ARTIFACTS_ROOT / level / name
        if out_dir.is_dir() and (out_dir / "kernel_code.py").is_file():
            print(f"Skip {level}/{name}: artifacts already exist.")
            continue

        # 调用 Fuser.pipeline
        cmd = [
            PYTHON_EXEC,
            "-m",
            "Fuser.pipeline",
            "--problem",
            str(problem_path.resolve()),
            "--extract-model",
            DEFAULT_EXTRACT_MODEL,
            "--dispatch-model",
            DEFAULT_DISPATCH_MODEL,
            "--dispatch-jobs",
            DEFAULT_DISPATCH_JOBS,
            "--compose-model",
            DEFAULT_COMPOSE_MODEL,
            "--workers",
            str(DEFAULT_WORKERS),
            "--max-iters",
            str(DEFAULT_MAX_ITERS),
            "--llm-timeout-s",
            str(DEFAULT_LLM_TIMEOUT_S),
            "--run-timeout-s",
            str(DEFAULT_RUN_TIMEOUT_S),
            "--target-platform",
            DEFAULT_TARGET_PLATFORM,
            "--verify",
        ]
        print("Running:", " ".join(cmd))
        rc, out, err = _run_cmd(cmd, cwd=KERNELAGENT_ROOT)

        if rc != 0:
            print(f"[FAIL] pipeline rc={rc} for {level}/{name}", file=sys.stderr)
            print("stdout:\n", out)
            print("stderr:\n", err)
            continue

        # Fuser.pipeline 会打印一个 JSON，里面有 run_dir 字段
        run_dir: Optional[Path] = None
        try:
            # 取 stdout 最后一行的 JSON
            out_stripped = out.strip().splitlines()[-1]
            res = json.loads(out_stripped)
            run_dir = Path(res["run_dir"]).resolve()
        except Exception:
            # 回退：到 .fuse 中找最新的 run
            run_dir = _find_latest_fuse_run()
            print("[WARN] failed to parse run_dir from stdout; fallback to latest .fuse run:", run_dir)

        if not run_dir or not run_dir.is_dir():
            print(f"[FAIL] Cannot locate run_dir for {level}/{name}", file=sys.stderr)
            continue

        composed_path, summary_data = _extract_artifacts_from_run(run_dir)
        if not composed_path:
            print(f"[FAIL] No composed_kernel.py under {run_dir}/compose_out for {level}/{name}", file=sys.stderr)
            continue

        _write_single_artifact_set(
            problem_path=problem_path,
            composed_kernel_path=composed_path,
            summary_data=summary_data,
            out_root=ARTIFACTS_ROOT,
        )
        break

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
