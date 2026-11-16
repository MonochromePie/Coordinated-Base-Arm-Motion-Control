"""
Plot the whole-body runs of the walkie-ros execution benchmark (full_body_ik/benchmark_execution.py)
for the final report (figures/execution_benchmark.png), and print the summary numbers of Table tab:exec
and the rows of the per-trial appendix table.

    uv run report/final_draft/scripts/plot_execution.py ~/walkie-ros/full_body_ik/results/execution_runs.csv
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE = "whole_body"
COLOR = "#1a9e77"
OUT = Path(__file__).resolve().parent.parent / "figures" / "execution_benchmark.png"
KEYS = [("time_sim_s", "Execution time", "s"), ("pos_err_mm", "Final hand position error (max of both hands)", "mm"),
        ("base_err_mm", "Final base position error", "mm")]


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["mode"] == MODE]
    for r in rows:
        for k in ("trial", "success", "time_sim_s", "pos_err_mm", "rot_err_deg", "base_err_mm",
                  "base_yaw_err_deg", "base_goal_dist_m"):
            r[k] = float(r[k])
    trials = [int(r["trial"]) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (key, title, unit) in zip(ax, KEYS):
        vals = [r[key] for r in rows]
        a.bar(trials, vals, color=COLOR)
        a.axhline(statistics.median(vals), color="k", lw=1, ls="--", label="median")
        a.set_title(title)
        a.set_xlabel("trial")
        a.set_xticks(trials[::2])
        a.set_ylabel(f"[{unit}]")
        a.grid(axis="y", alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")

    print(f"\nsucceeded {int(sum(r['success'] for r in rows))}/{len(rows)}")
    for key in ("time_sim_s", "pos_err_mm", "rot_err_deg", "base_err_mm", "base_yaw_err_deg"):
        v = [r[key] for r in rows]
        print(f"{key:18s} median {statistics.median(v):.2f}  min {min(v):.2f}  max {max(v):.2f}")

    print("\nappendix rows:")
    for r in rows:
        print(f"        {int(r['trial'])} & {r['base_goal_dist_m']:.2f} & {r['time_sim_s']:.1f} & {r['pos_err_mm']:.1f} & "
              f"{r['rot_err_deg']:.2f} & {r['base_err_mm']:.1f} \\\\")


if __name__ == "__main__":
    main()
