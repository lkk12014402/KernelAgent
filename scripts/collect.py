#!/usr/bin/env python3
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

KERNELAGENT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = KERNELAGENT_ROOT / "artifacts"


def infer_level_and_name(problem_path: Path) -> Tuple[str, str]:
    level_dir = problem_path.parent.name  # e.g. "level1"
    m = re.match(r"level(\d+)", level_dir)
    level = f"L{m.group(1)}" if m else level_dir
    name = problem_path.stem  # e.g. "19_ReLU"
    return level, name


def find_latest_fuse_run(base: Path) -> Optional[Path]:
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_artifacts_from_run(run_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
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


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: collect_artifacts.py PROBLEM_PATH [RUN_DIR_FROM_STDOUT_JSON]", file=sys.stderr)
        return 1

    problem_path = Path(argv[1]).resolve()
    if not problem_path.is_file():
        print(f"problem not found: {problem_path}", file=sys.stderr)
        return 1

    run_dir: Optional[Path] = None
    if len(argv) >= 3 and argv[2]:
        # 第二个参数是 Fuser.pipeline stdout 最后一行 JSON 里的 run_dir
        try:
            run_dir = Path(argv[2]).resolve()
        except Exception:
            run_dir = None

    if run_dir is None or not run_dir.is_dir():
        # 回退：从 .fuse 中找最新的 run
        base = Path.cwd() / ".fuse"
        run_dir = find_latest_fuse_run(base)
        if run_dir is None:
            print("No .fuse run dir found", file=sys.stderr)
            return 1

    composed, summary_data = extract_artifacts_from_run(run_dir)
    if not composed:
        print(f"No composed_kernel.py under {run_dir}/compose_out", file=sys.stderr)
        return 1

    level, name = infer_level_and_name(problem_path)
    out_dir = ARTIFACTS_ROOT / level / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # original.py
    original_dst = out_dir / "original.py"
    shutil.copy2(problem_path, original_dst)

    # kernel_code.py
    kernel_dst = out_dir / "kernel_code.py"
    shutil.copy2(composed, kernel_dst)

    # manifest.json
    manifest: Dict[str, Any] = {
        "level": level,
        "name": name,
        "files": {
            "original": "original.py",
            "kernel": "kernel_code.py",
        },
        "route": "pipeline",
        "verify": {},
    }
    if summary_data:
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

    print(f"[OK] artifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
