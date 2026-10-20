#!/usr/bin/env python3
"""Run an isolated 300k-step height/sway experiment for seeds 8 through 100."""

from __future__ import annotations

import concurrent.futures
import csv
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CHECKPOINTS = HERE / "checkpoints"
SWEEP_DIR = CHECKPOINTS / "seed_sweep_8_100"
SEEDS = range(8, 101)
MAX_PARALLEL_RUNS = 3
FINAL_RE = re.compile(
    r"Final deterministic policy: (\d+)/20 full stands; "
    r"mean planted-foot sway=([0-9.]+) m/s; mean reward=([-0-9.]+)"
)


def run_seed(seed: int) -> dict[str, str | int]:
    run_name = f"prone_height_sway_balanced_10_seed{seed}_sweep"
    run_dir = CHECKPOINTS / run_name
    log_path = SWEEP_DIR / f"seed{seed}.log"
    command = [
        sys.executable,
        str(HERE / "train_height_sway.py"),
        "--timesteps",
        "300000",
        "--n-envs",
        "8",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = FINAL_RE.findall(log_text)
    row: dict[str, str | int] = {
        "seed": seed,
        "exit_code": result.returncode,
        "successes_out_of_20": "",
        "mean_planted_foot_sway_m_s": "",
        "mean_reward": "",
        "run_dir": str(run_dir),
        "log_path": str(log_path),
    }
    if matches:
        row["successes_out_of_20"] = matches[-1][0]
        row["mean_planted_foot_sway_m_s"] = matches[-1][1]
        row["mean_reward"] = matches[-1][2]
    return row


def main() -> None:
    run_names = [
        f"prone_height_sway_balanced_10_seed{seed}_sweep" for seed in SEEDS
    ]
    existing_runs = [name for name in run_names if (CHECKPOINTS / name).exists()]
    if existing_runs:
        raise SystemExit(f"Refusing to overwrite existing run folder(s): {existing_runs}")
    SWEEP_DIR.mkdir(parents=True, exist_ok=False)
    summary_path = SWEEP_DIR / "summary.csv"
    fieldnames = [
        "seed",
        "exit_code",
        "successes_out_of_20",
        "mean_planted_foot_sway_m_s",
        "mean_reward",
        "run_dir",
        "log_path",
    ]
    completed_rows: list[dict[str, str | int]] = []
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        summary_file.flush()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_RUNS
        ) as executor:
            futures = {executor.submit(run_seed, seed): seed for seed in SEEDS}
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                try:
                    row = future.result()
                except Exception as error:  # keep the remaining sweep running
                    row = {
                        "seed": seed,
                        "exit_code": -1,
                        "successes_out_of_20": "",
                        "mean_planted_foot_sway_m_s": "",
                        "mean_reward": "",
                        "run_dir": str(
                            CHECKPOINTS
                            / f"prone_height_sway_balanced_10_seed{seed}_sweep"
                        ),
                        "log_path": f"ERROR: {error}",
                    }
                completed_rows.append(row)
                writer.writerow(row)
                summary_file.flush()
                print(
                    f"Finished seed {seed}: exit={row['exit_code']}, "
                    f"successes={row['successes_out_of_20']}/20; "
                    f"{len(completed_rows)}/{len(SEEDS)} runs complete",
                    flush=True,
                )

    successes = sorted(
        int(row["seed"])
        for row in completed_rows
        if str(row["successes_out_of_20"]).isdigit()
        and int(str(row["successes_out_of_20"])) > 0
    )
    perfect = sorted(
        int(row["seed"])
        for row in completed_rows
        if str(row["successes_out_of_20"]) == "20"
    )
    print(f"Sweep complete. Results: {summary_path}", flush=True)
    print(f"Seeds with at least one success: {successes}", flush=True)
    print(f"Seeds with 20/20 successes: {perfect}", flush=True)


if __name__ == "__main__":
    main()
