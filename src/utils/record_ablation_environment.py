#!/usr/bin/env python3
"""Record software, GPU, revision, and measured task runtimes."""

from pathlib import Path
import platform
import re
import subprocess

import accelerate
import diffusers
import peft
import torch
import transformers


def command(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return (result.stdout + result.stderr).strip()


def runtime(path: Path) -> str:
    if not path.is_file():
        return "N/A"
    match = re.findall(r"runtime_seconds=(\d+)", path.read_text())
    return match[-1] if match else "N/A"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    tasks = {
        "A": root / "logs/ablations/task_a_ip_adapter/runtime.txt",
        "B": root / "logs/ablations/task_b_rank/runtime.txt",
        "C": root / "logs/ablations/task_c_steps/runtime.txt",
        "D": root / "logs/ablations/task_d_face_corruption/runtime.txt",
        "E": root / "logs/ablations/task_e_refcount/runtime.txt",
        "F": root / "logs/ablations/task_f_prior/runtime.txt",
    }
    text = [
        "# Ablation environment and runtime report", "",
        f"- Python: {platform.python_version()}", f"- torch: {torch.__version__}",
        f"- torch CUDA: {torch.version.cuda}", f"- diffusers: {diffusers.__version__}",
        f"- transformers: {transformers.__version__}", f"- accelerate: {accelerate.__version__}",
        f"- peft: {peft.__version__}",
        f"- Git commit: {command(['git', 'rev-parse', 'HEAD'], root)}", "",
        "## Measured task runtimes", "", "| Task | Runtime (seconds) |", "|---|---:|",
    ]
    text.extend(f"| {task} | {runtime(path)} |" for task, path in tasks.items())
    text.extend(["", "## GPU information", "", "```text", command(["nvidia-smi", "-L"], root), "```", "",
                 "## Git status (abbreviated)", "", "```text",
                 command(["git", "status", "--short"], root), "```", ""])
    destination = root / "results/system/ablation_environment_report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(text))
    diff_path = root / "results/system/ablation_git_diff.patch"
    diff_path.write_text(command(["git", "diff", "--", ".", ":(exclude)results/system/ablation_git_diff.patch"], root) + "\n")
    print(f"ENVIRONMENT_REPORT_COMPLETE path={destination} diff={diff_path}")


if __name__ == "__main__":
    main()
