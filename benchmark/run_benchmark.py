"""
IK benchmark: how the solver evolved, measured.

Compares the IK versions in walker_robot/ik_versions.py (stages 1-5) on the scenarios in
walker_robot/scenarios.py:

    A  Singularity demo (stage 1 vs 2), from a nearly straight left arm (elbow q_l4 = 0.05 rad):
         A1 raise the hand 10 cm (reachable, but the start is close to a singularity)
         A2 push the hand 10 cm further down (out of reach: the arm stretches into the singularity)
    B  Rest weight (stage 3 vs 4 vs 5) on one bimanual target that needs the base and lift
    C  Random reachable targets for stages 1-5: convergence rate, iterations, solve time, accuracy

Every method is iterated q <- clip(q + dq, limits) from a fresh start until the error is under the
tolerance or max_iters is reached. There is no per-step cap (BimanualIK.MAX_JOINT_STEP) here, so the
numbers describe the solver itself rather than the speed limit of the controller.
To SEE these runs on the robot, use ik_sim/run_ik_sim.py.

Run (from the repo root):
      uv run benchmark/run_benchmark.py                    (all experiments)
      uv run benchmark/run_benchmark.py --only C --n 500   (one experiment, more targets)
Outputs: tables on stdout; CSV + PNG plots in benchmark/results/ (plots need matplotlib).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "walker_robot"))

import argparse
import csv

import numpy as np

from bimanual_ik import LEFT_ARM_SYMBOLS, lift
from ik_versions import BASE_ALL, METHODS, Arms, run, select_targets, vec
from scenarios import READY, SCENARIOS, pose_targets, random_config, scenario

OUT_DIR = Path(__file__).resolve().parent / "results"
STAGES = ("pinv", "dls", "rest_left", "rest_both", "current")    # experiment C


# =====================================================================================
# Experiment A: singularity
# =====================================================================================

def experiment_singularity(arms, args):
    print("\n" + "=" * 90)
    print("A. SINGULARITY: pseudoinverse vs damped least squares (left arm, position only, base locked)")
    print("=" * 90)
    q0, _ = scenario(arms, "A1")
    s = np.linalg.svd(arms.task(q0, {"left": (np.zeros(3), None)}, LEFT_ARM_SYMBOLS)[0], compute_uv=False)
    print(f"start: q_l4 = 0.05 rad, singular values of J = {np.array2string(s, precision=4)}, "
          f"condition number = {s[0] / s[-1]:.0f}")

    keys = ("A1", "A2")
    results = {}
    rows = []
    print(f"\n{'scenario':<42}{'method':<30}{'conv':>5}{'iters':>7}{'pos err [mm]':>14}"
          f"{'err after 1st [mm]':>20}{'1st |dq| [rad]':>16}{'joint travel [rad]':>20}")
    for key in keys:
        q0, targets = scenario(arms, key)
        for mk in ("pinv", "dls"):
            m = METHODS[mk]
            r = run(arms, m, q0, select_targets(m, targets), args.pos_tol, args.rot_tol, args.sing_iters, record=True)
            results[(key, mk)] = r
            first = r.step_hist[0] if r.step_hist else float("nan")
            travel = np.linalg.norm(vec(r.q_final, m.solve) - vec(q0, m.solve))
            print(f"{SCENARIOS[key]:<42}{m.label:<30}{'yes' if r.converged else 'NO':>5}{r.iters:>7}"
                  f"{r.pos_err * 1e3:>14.3f}{r.pos_hist[1] * 1e3:>20.1f}{first:>16.3f}{travel:>20.3f}")
            for i, (pe, st) in enumerate(zip(r.pos_hist, [np.nan] + r.step_hist)):
                rows.append([key, mk, i, pe, st])
    print("\nNotes: A1 has a solution, A2's best answer is 10 cm away (the closest reachable point).")
    print("       |dq| is the largest joint step of the first iteration, before clipping to the joint limits.")
    print("       joint travel = ||q_final - q_start||: how far the solution ended from where the arm started.")
    write_csv("A_singularity.csv", ["scenario", "method", "iter", "pos_err_m", "max_abs_dq"], rows)
    if args.plots:
        plot_singularity(results, keys)


# =====================================================================================
# Experiment B: rest weight vs null space (one bimanual target)
# =====================================================================================

def experiment_rest(arms, args):
    print("\n" + "=" * 90)
    print("B. REST WEIGHT: one arm vs two arms vs current null-space solver")
    print("=" * 90)
    print("target: asymmetric pose of both hands, from a configuration with base x = 0.1 m, yaw = 0.3 rad, "
          "lift = 0.4 m")
    print("start: both arms at the rest pose, lift 0.2 m")
    print(f"tolerance: position < {args.pos_tol * 1e3:g} mm, orientation < {np.degrees(args.rot_tol):.2f} deg")
    q0, targets = scenario(arms, "B")
    results = {}
    rows = []
    print(f"\n{'method':<48}{'conv':>5}{'iters':>7}{'L err [mm]':>12}{'R err [mm]':>12}{'rot err [deg]':>15}"
          f"{'lift [m]':>10}{'time [ms]':>11}")
    for mk in ("rest_left", "rest_both", "current"):
        m = METHODS[mk]
        t = select_targets(m, targets)
        r = run(arms, m, q0, t, args.pos_tol, args.rot_tol, args.rest_iters, record=True)
        side_hist = [arms.side_errors(q, t) for q in r.qs]
        results[mk] = (r, side_hist)
        err = side_hist[-1]
        r_txt = f"{err['right'] * 1e3:.3f}" if "right" in err else "-"
        print(f"{m.label:<48}{'yes' if r.converged else 'NO':>5}{r.iters:>7}{err['left'] * 1e3:>12.3f}{r_txt:>12}"
              f"{np.degrees(r.rot_err):>15.3f}{r.q_final[lift]:>10.3f}{r.time_s * 1e3:>11.1f}")
        for i, (e, re) in enumerate(zip(side_hist, r.rot_hist)):
            rows.append([mk, i, e["left"], e.get("right", ""), re])
    print("\nThe rest weight is a cost the solver trades against the target error. With one arm the")
    print("leftover error is small; with two arms the shared base/lift is pulled by both tasks and the")
    print("rest pose at once, and the error settles above the tolerance.")
    print("The current solver puts the rest pull in the null space of J, where it cannot move the hands.")
    write_csv("B_rest_weight.csv", ["method", "iter", "left_pos_err_m", "right_pos_err_m", "rot_err_rad"], rows)
    if args.plots:
        plot_rest(results, args)


# =====================================================================================
# Experiment C: random targets
# =====================================================================================

def experiment_random(arms, args):
    print("\n" + "=" * 90)
    print(f"C. RANDOM TARGETS: {args.n} reachable targets (FK of random configurations), seed {args.seed}")
    print(f"   tolerance: position < {args.pos_tol * 1e3:g} mm, orientation < {np.degrees(args.rot_tol):.2f} deg, "
          f"max {args.max_iters} iterations")
    print("=" * 90)
    print("Stages 1-2 lock the base, so their targets come from random ARM joints only (base at the start pose).")
    rng = np.random.default_rng(args.seed)
    configs = [random_config(rng) for _ in range(args.n)]

    runs = {}
    rows = []
    for mk in STAGES:
        m = METHODS[mk]
        rs = []
        for i, qc in enumerate(configs):
            if not any(s in m.solve for s in BASE_ALL):
                qc = {**qc, **{s: READY[s] for s in BASE_ALL}}    # base locked: target must not need it
            r = run(arms, m, READY, select_targets(m, pose_targets(arms, qc)), args.pos_tol, args.rot_tol,
                    args.max_iters, record=args.plots)
            rs.append(r)
            rows.append([mk, i, int(r.converged), r.iters, r.pos_err, r.rot_err, r.time_s, r.max_step, r.failed])
        runs[mk] = rs
        print(f"  {m.label}: done")

    print(f"\n{'method':<48}{'conv %':>7}{'iters':>13}{'ms/iter':>9}{'solve ms':>15}"
          f"{'pos err mm':>17}{'rot err deg':>17}{'max|dq|':>9}")
    print(f"{'':<48}{'':>7}{'med (p95)':>13}{'med':>9}{'med (p95)':>15}{'med (p95)':>17}{'med (p95)':>17}"
          f"{'p95':>9}")
    summary = []
    for mk, rs in runs.items():
        m = METHODS[mk]
        conv = [r for r in rs if r.converged]
        rate = 100.0 * len(conv) / len(rs)
        it = np.array([r.iters for r in conv]) if conv else np.array([np.nan])
        t = np.array([r.time_s for r in conv]) * 1e3 if conv else np.array([np.nan])
        per_iter = np.array([r.time_s / r.iters for r in rs if r.iters > 0]) * 1e3
        pe = np.array([r.pos_err for r in rs]) * 1e3
        re = np.degrees(np.array([r.rot_err for r in rs]))
        ms = np.array([r.max_step for r in rs])
        rot_txt = f"{np.median(re):.3f} ({np.percentile(re, 95):.3f})" if m.orientation else "-"
        print(f"{m.label:<48}{rate:>7.1f}{fmt_mp(it, '.0f'):>13}{np.median(per_iter):>9.3f}{fmt_mp(t, '.1f'):>15}"
              f"{fmt_mp(pe, '.3f'):>17}{rot_txt:>17}{np.percentile(ms, 95):>9.2f}")
        summary.append([mk, m.label, rate, np.median(it), np.percentile(it, 95), np.median(per_iter),
                        np.median(t), np.percentile(t, 95), np.median(pe), np.percentile(pe, 95),
                        np.median(re) if m.orientation else "", np.percentile(re, 95) if m.orientation else "",
                        np.percentile(ms, 95)])
        fails = [r.failed for r in rs if r.failed]
        if fails:
            print(f"{'':<48}  {len(fails)} runs aborted: {', '.join(sorted(set(fails)))}")
    print("\niters / solve ms: over converged runs only. pos/rot err: final error of every run.")
    print("ms/iter includes evaluating FK and the Jacobian (lambdified sympy), which dominates.")

    write_csv("C_random_runs.csv", ["method", "target", "converged", "iters", "pos_err_m", "rot_err_rad",
                                    "time_s", "max_abs_dq", "failed"], rows)
    write_csv("C_random_summary.csv", ["method", "label", "conv_rate_pct", "iters_median", "iters_p95",
                                       "ms_per_iter_median", "solve_ms_median", "solve_ms_p95",
                                       "pos_err_mm_median", "pos_err_mm_p95", "rot_err_deg_median",
                                       "rot_err_deg_p95", "max_abs_dq_p95"], summary)
    if args.plots:
        plot_random(runs, args)


def fmt_mp(a, f):
    if np.all(np.isnan(a)):
        return "-"
    return f"{np.median(a):{f}} ({np.percentile(a, 95):{f}})"


def write_csv(name, header, rows):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> {OUT_DIR / name}")


# =====================================================================================
# Plots
# =====================================================================================

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })


def _save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    print(f"  -> {OUT_DIR / name}")


def plot_singularity(results, keys):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, key in enumerate(keys):
        for mk in ("pinv", "dls"):
            r, m = results[(key, mk)], METHODS[mk]
            axes[0, col].semilogy(range(len(r.pos_hist)), np.maximum(r.pos_hist, 1e-12), color=m.color, label=m.label)
            axes[1, col].semilogy(range(1, len(r.step_hist) + 1), np.maximum(r.step_hist, 1e-12),
                                  color=m.color, label=m.label)
        axes[0, col].set_title(SCENARIOS[key])
        axes[0, col].set_ylabel("position error [m]")
        axes[1, col].set_ylabel("largest joint step |dq| [rad]")
        axes[1, col].set_xlabel("iteration")
    axes[0, 0].legend()
    fig.suptitle("Pseudoinverse vs DLS from a nearly straight arm (q_l4 = 0.05 rad)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "A_singularity.png")
    plt.close(fig)


def plot_rest(results, args):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for mk, (r, side_hist) in results.items():
        m = METHODS[mk]
        for side, ls in (("left", "-"), ("right", "--")):
            if side in m.sides:
                axes[0].semilogy([max(e[side], 1e-12) for e in side_hist], color=m.color, linestyle=ls,
                                 label=f"{m.label}, {side} hand")
        axes[1].semilogy(np.maximum(np.degrees(r.rot_hist), 1e-12), color=m.color, label=m.label)
    axes[0].axhline(args.pos_tol, color=INK2, linestyle=":", linewidth=1)
    axes[0].text(1, args.pos_tol * 1.15, "tolerance", color=INK2, fontsize=9)
    axes[1].axhline(np.degrees(args.rot_tol), color=INK2, linestyle=":", linewidth=1)
    axes[0].set_ylabel("position error [m]")
    axes[1].set_ylabel("worst-arm orientation error [deg]")
    for ax in axes:
        ax.set_xlabel("iteration")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rest weight leaves a steady-state error with two arms; the null-space rest pull does not",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "B_rest_weight.png")
    plt.close(fig)


def plot_random(runs, args):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # convergence: fraction of targets solved by iteration k
    for mk, rs in runs.items():
        m = METHODS[mk]
        its = np.sort([r.iters for r in rs if r.converged])
        k = np.arange(0, args.max_iters + 1)
        axes[0].plot(k, 100.0 * np.searchsorted(its, k, side="right") / len(rs), color=m.color, label=m.label)
    axes[0].set_xscale("symlog", linthresh=10)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("targets solved [%]")
    axes[0].set_title("Convergence rate")
    axes[0].legend(fontsize=8, loc="upper left")

    # median error vs iteration (runs that stopped early hold their final error)
    for mk, rs in runs.items():
        m = METHODS[mk]
        L = args.max_iters + 1
        H = np.array([np.pad(r.pos_hist, (0, L - len(r.pos_hist)), mode="edge") for r in rs])
        axes[1].semilogy(np.maximum(np.median(H, axis=0), 1e-12), color=m.color, label=m.label)
    axes[1].axhline(args.pos_tol, color=INK2, linestyle="--", linewidth=1)
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("median position error [m]")
    axes[1].set_title("Error vs iteration")

    # final accuracy: CDF of final position error
    for mk, rs in runs.items():
        m = METHODS[mk]
        pe = np.sort([max(r.pos_err, 1e-9) for r in rs])
        axes[2].plot(pe, 100.0 * np.arange(1, len(pe) + 1) / len(pe), color=m.color, label=m.label)
    axes[2].axvline(args.pos_tol, color=INK2, linestyle="--", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("final position error [m]")
    axes[2].set_ylabel("targets [%]")
    axes[2].set_title("Final accuracy (CDF)")

    fig.tight_layout()
    _save(fig, "C_random.png")
    plt.close(fig)


# =====================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["A", "B", "C"], nargs="+", default=["A", "B", "C"])
    ap.add_argument("--n", type=int, default=200, help="number of random targets (experiment C)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-tol", type=float, default=1e-3, help="position tolerance [m]")
    ap.add_argument("--rot-tol", type=float, default=np.radians(0.5), help="orientation tolerance [rad]")
    ap.add_argument("--max-iters", type=int, default=300, help="iteration limit (experiment C)")
    ap.add_argument("--sing-iters", type=int, default=100, help="iteration limit (experiment A)")
    ap.add_argument("--rest-iters", type=int, default=300, help="iteration limit (experiment B)")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    args = ap.parse_args()

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            print("matplotlib not installed: skipping plots (uv sync --dev)")
            args.plots = False

    arms = Arms()
    if "A" in args.only:
        experiment_singularity(arms, args)
    if "B" in args.only:
        experiment_rest(arms, args)
    if "C" in args.only:
        experiment_random(arms, args)


if __name__ == "__main__":
    main()
