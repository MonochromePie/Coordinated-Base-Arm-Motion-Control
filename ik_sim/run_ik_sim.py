"""
IK playback in MuJoCo: any IK version (walker_robot/ik_versions.py) on any test scenario
(walker_robot/scenarios.py), shown on the robot. The visual side of benchmark/run_benchmark.py.

Scenarios (default methods in brackets):
    A1  singularity: raise the left hand 10 cm from a nearly straight arm        [pinv dls current_arm current]
    A2  singularity: push the left hand 10 cm down, out of reach                [pinv dls current_arm current]
    B   rest weight: both hands to an asymmetric 6-D target from the rest pose  [rest_left rest_both current]

IK versions (--methods, any of them on any scenario; a method only gets the targets it tracks):
    pinv         1. Jacobian pseudoinverse, left arm, position only
    dls          2. Damped least squares, left arm, position only
    rest_left    3. DLS + rest weight, left arm + base
    rest_both    4. DLS + rest weight, both arms + base
    current      5. Current solver (null-space rest), both arms + base
    current_arm  5. Current solver, left arm only, position only

Each method is iterated q <- clip(q + dq, limits) as in the benchmark, and the joint states are
shown KINEMATICALLY (qpos set directly, no physics), so what you see is exactly the IK output.
--capped adds BimanualIK's per-step cap (MAX_JOINT_STEP), i.e. what the controller would send.

Viewer: red spheres are the targets, grey ones the start, the coloured dots the hand paths.
    space = pause/resume,  N = next method,  R = restart this method

Run (from the repo root):
      uv run ik_sim/run_ik_sim.py                              (viewer, A1, its default methods in turn)
      uv run ik_sim/run_ik_sim.py B                            (scenario B)
      uv run ik_sim/run_ik_sim.py A2 --methods pinv dls
      uv run ik_sim/run_ik_sim.py B --record                   (offscreen: side-by-side GIF + keyframe strip
                                                                in ik_sim/results/, for the report)
      uv run ik_sim/run_ik_sim.py A2 --record --capped
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "walker_robot"))

import argparse
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from bimanual_ik import JOINT_NAMES, BimanualIK
from ik_versions import METHODS, Arms, run, select_targets
from scenarios import SCENARIOS, scenario
from sim import IK_TO_SIM_OFFSET, MujocoSim

SCENE = str(ROOT / "simulation" / "walker_scene.xml")
OUT_DIR = Path(__file__).resolve().parent / "results"
POS_TOL, ROT_TOL = 1e-3, np.radians(0.5)
# playback pacing: frames follow joint MOTION, not iterations, so tiny steps don't look like a frozen robot
RAD_PER_FRAME = 0.015   # largest joint change shown per frame (rad, or m for x/y/lift) = on-screen speed
MAX_FRAMES_PER_ITER = 12    # a huge step (e.g. the pseudoinverse at a singularity) still plays in <= 0.4 s
MAX_SKIP = 10           # iterations that barely move are fast-forwarded, at most this many per frame
MAX_FRAMES = 300        # per method (10 s at 30 fps): a run that keeps moving (pinv flailing) plays faster


@dataclass
class View:
    methods: tuple          # shown by default
    iters: int              # default iteration limit (without --capped)
    key_iters: tuple        # iterations shown as rows of the keyframe strip (plus the last one)
    start_note: str         # for the keyframe strip title
    azimuth: float
    elevation: float
    distance: float
    focus: str              # camera look-at: "left_shoulder" or "chest"
    show_smin: bool         # print the left arm's smallest singular value


VIEWS = {
    # camera on the left arm from the robot's left-front, far enough to keep an arm thrown overhead in view
    "A1": View(("pinv", "dls", "current_arm", "current"), 100, (0, 1, 2, 5, 20),
               "start: elbow q_l4 = 0.05 rad", -115.0, -8.0, 2.1, "left_shoulder", True),
    "A2": View(("pinv", "dls", "current_arm", "current"), 100, (0, 1, 2, 5, 20),
               "start: elbow q_l4 = 0.05 rad", -115.0, -8.0, 2.1, "left_shoulder", True),
    # camera in front of the robot, slightly to its left, looking at the chest
    "B": View(("rest_left", "rest_both", "current"), 300, (0, 2, 10, 50),
              "start: both arms at the rest pose", 155.0, -15.0, 2.3, "chest", False),
}


def trajectory(arms, method, q0, targets, iters, cap, show_smin):
    """
    Iterate the method; returns (list of joint states, list of per-state info dicts).
    info: iteration, position error per side [m], raw largest step |dq| that produced this state,
    sigma_min of the left arm (if show_smin), status.
    """
    r = run(arms, method, q0, targets, POS_TOL, ROT_TOL, iters, cap=cap, record=True)
    info = [dict(it=i, err=arms.side_errors(q, targets), step=([0.0] + r.step_hist)[i],
                 smin=arms.sigma_min(q) if show_smin else None, status="")
            for i, q in enumerate(r.qs)]
    info[-1]["status"] = ("converged" if r.converged else r.failed or f"NOT converged after {iters} iterations")
    return r.qs, info


def pace(qs, info, rad_per_frame, max_per_iter=MAX_FRAMES_PER_ITER):
    """
    pace_once, sped up (fewer frames per big step, then more rad per frame) until it fits MAX_FRAMES.
    """
    frames = pace_once(qs, info, rad_per_frame, max_per_iter)
    while len(frames) > MAX_FRAMES and max_per_iter > 1:
        max_per_iter = max(1, max_per_iter // 2)
        rad_per_frame *= 2
        frames = pace_once(qs, info, rad_per_frame, max_per_iter)
    return frames


def pace_once(qs, info, rad_per_frame, max_per_iter):
    """
    Animation frames, linear in joint space from the last shown state to the next one, with as many
    frames as the motion needs at rad_per_frame (at most max_per_iter). Iterations that move less than
    rad_per_frame are skipped (up to MAX_SKIP in a row, so the iteration counter keeps running through
    a plateau). Each frame carries the info of the state it moves towards.
    """
    frames = [(qs[0], info[0])]
    shown, skipped = qs[0], 0
    for k in range(1, len(qs)):
        b, inf = qs[k], info[k]
        move = max(abs(b[s] - shown[s]) for s in b)
        if move < rad_per_frame and skipped < MAX_SKIP - 1 and k < len(qs) - 1:
            skipped += 1
            continue
        n = max(1, min(int(np.ceil(move / rad_per_frame)), max_per_iter))
        for j in range(1, n + 1):
            t = j / n
            frames.append(({s: (1 - t) * shown[s] + t * b[s] for s in b}, inf))
        shown, skipped = b, 0
    return frames


def to_names(q):
    return {s.name: float(v) for s, v in q.items()}


def hex_rgba(color, alpha=1.0):
    return np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [alpha], np.float32)


def add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, float), np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def decorate(scn, starts, targets, trails, color):
    for side, (tp, _) in targets.items():
        add_sphere(scn, starts[side], 0.018, np.array([0.7, 0.7, 0.7, 0.6], np.float32))
        add_sphere(scn, tp + IK_TO_SIM_OFFSET, 0.022, np.array([0.9, 0.1, 0.1, 0.8], np.float32))
    rgba = hex_rgba(color)
    for trail in trails.values():
        for p in trail[-400:]:
            add_sphere(scn, p, 0.006, rgba)


def make_camera(view, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = view.azimuth, view.elevation, view.distance
    return cam


def caption(inf, long=False):
    errs = "  ".join(f"{side[0].upper()} {e * 1e3:.1f} mm" for side, e in inf["err"].items())
    line = f"iter {inf['it']}   {errs}   |dq| {inf['step']:.2f} rad"
    if long and inf["smin"] is not None:
        line += f"   sigma_min {inf['smin']:.3f}"
    if long and inf["status"]:
        line += f"   [{inf['status']}]"
    return line


# =====================================================================================
# Interactive viewer: methods one after another
# =====================================================================================

def run_viewer(sim, runs, starts, view, lookat, args):
    import mujoco.viewer

    keys = {"pause": False, "next": False, "restart": False}

    def on_key(code):
        if code == 32:
            keys["pause"] = not keys["pause"]
        elif code == ord("N"):
            keys["next"] = True
        elif code == ord("R"):
            keys["restart"] = True

    dt = 1.0 / args.fps
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
        with viewer.lock():
            cam = make_camera(view, lookat)
            viewer.cam.lookat[:], viewer.cam.azimuth = cam.lookat, cam.azimuth
            viewer.cam.elevation, viewer.cam.distance = cam.elevation, cam.distance
        k = 0
        while viewer.is_running():
            method, targets, frames = runs[k % len(runs)]
            print(f"\n--- {method.label}: {len(frames)} frames ---")
            trails = {side: [] for side in targets}
            i = 0
            while viewer.is_running() and i < len(frames) + int(args.hold * args.fps):
                t0 = time.perf_counter()
                if keys["next"] or keys["restart"]:
                    break
                q, inf = frames[min(i, len(frames) - 1)]
                if i < len(frames):
                    sim.set_joint_state(to_names(q))
                    for side in trails:
                        trails[side].append(sim.tcp_pose(side)[0])
                    if i == 0 or inf is not frames[i - 1][1]:
                        print("  " + caption(inf, long=True))
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    decorate(viewer.user_scn, starts, targets, trails, method.color)
                # NOT inside viewer.lock(): set_texts takes the viewer's own locks, and calling it with
                # lock() held deadlocks the next sync() now and then (the viewer "freezes")
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  method.label, caption(inf, long=True)))
                viewer.sync()
                if not keys["pause"]:
                    i += 1
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            if keys["restart"]:
                keys["restart"] = False
            else:
                keys["next"] = False
                k += 1


# =====================================================================================
# Offscreen recording: side-by-side GIF + keyframe strip
# =====================================================================================

def record(sim, runs, starts, view, lookat, args):
    from PIL import Image, ImageDraw, ImageFont

    W, H = args.size
    renderer = mujoco.Renderer(sim.model, H, W)
    cam = make_camera(view, lookat)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = font_b = ImageFont.load_default()

    n_frames = max(len(f) for _, _, f in runs)
    trails = [{side: [] for side in targets} for _, targets, _ in runs]
    gif = []
    # keyframe strip: row j = key iteration j (then the final state), each panel at ITS OWN frame where it
    # reaches that iteration (the GIF is paced by motion, so the methods reach it at different frames)
    key_frames = [[next((i for i, (_, inf) in enumerate(f) if inf["it"] >= it), len(f) - 1)
                   for it in view.key_iters] + [len(f) - 1] for _, _, f in runs]
    key_tiles = [[None] * (len(view.key_iters) + 1) for _ in runs]
    for i in range(n_frames):
        tiles = []
        for r, (method, targets, frames) in enumerate(runs):
            q, inf = frames[min(i, len(frames) - 1)]
            sim.set_joint_state(to_names(q))
            if i < len(frames):
                for side in trails[r]:
                    trails[r][side].append(sim.tcp_pose(side)[0])
            renderer.update_scene(sim.data, cam)
            decorate(renderer.scene, starts, targets, trails[r], method.color)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, W, 50], fill=(255, 255, 255))
            d.rectangle([0, 0, 6, 50], fill=method.color)
            d.text((14, 6), method.label, fill=(11, 11, 11), font=font_b)
            d.text((14, 28), caption(inf), fill=(82, 81, 78), font=font)
            if inf["status"] and i >= len(frames) - 1:
                d.text((14, H - 22), inf["status"], fill=(11, 11, 11), font=font_b)
            tiles.append(img)
            for j, kf in enumerate(key_frames[r]):
                if kf == min(i, len(frames) - 1) and key_tiles[r][j] is None:
                    key_tiles[r][j] = img
        row = Image.new("RGB", (W * len(tiles), H))
        for c, t in enumerate(tiles):
            row.paste(t, (c * W, 0))
        gif.append(row)

    stem = f"sim_{args.scenario}{'_capped' if args.capped else ''}"
    OUT_DIR.mkdir(exist_ok=True)
    # one fixed palette and no dithering: the static background is then identical in every frame and
    # the GIF only stores the pixels that change (dithering noise makes it ~20x bigger)
    palette = gif[-1].quantize(colors=128, dither=Image.Dither.NONE)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in gif]
    hold = [gif[-1]] * int(args.hold * args.fps)
    (gif + hold)[0].save(OUT_DIR / f"{stem}.gif", save_all=True, append_images=(gif + hold)[1:],
                         duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"  -> {OUT_DIR / stem}.gif")

    # keyframe strip: rows = moments, columns = methods
    rows = []
    for j in range(len(view.key_iters) + 1):
        row = Image.new("RGB", (W * len(runs), H))
        for c in range(len(runs)):
            row.paste(key_tiles[c][j], (c * W, 0))
        rows.append(row)
    head = 34
    strip = Image.new("RGB", (rows[0].width, head + H * len(rows)), (252, 252, 251))
    d = ImageDraw.Draw(strip)
    d.text((10, 8), f"{SCENARIOS[args.scenario]}   ({view.start_note})", fill=(11, 11, 11), font=font_b)
    for j, r in enumerate(rows):
        strip.paste(r, (0, head + j * H))
    strip.save(OUT_DIR / f"{stem}_keyframes.png")
    print(f"  -> {OUT_DIR / stem}_keyframes.png")


# =====================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", choices=list(SCENARIOS), default="A1")
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), default=None,
                    help="IK versions to show (default: the scenario's own set)")
    ap.add_argument("--capped", action="store_true",
                    help=f"cap each step at BimanualIK.MAX_JOINT_STEP = {BimanualIK.MAX_JOINT_STEP} (controller behaviour)")
    ap.add_argument("--iters", type=int, default=None, help="iteration limit (default per scenario, or 300 with --capped)")
    ap.add_argument("--speed", type=float, default=RAD_PER_FRAME,
                    help="playback speed: largest joint change per frame [rad] (with --capped: one frame per iteration)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final pose")
    ap.add_argument("--record", action="store_true", help="render offscreen to GIF + PNG instead of the viewer")
    ap.add_argument("--size", type=int, nargs=2, default=[440, 440], metavar=("W", "H"), help="panel size (--record)")
    args = ap.parse_args()
    view = VIEWS[args.scenario]
    args.methods = args.methods or list(view.methods)
    args.iters = args.iters or (300 if args.capped else view.iters)
    cap = BimanualIK.MAX_JOINT_STEP if args.capped else None

    arms = Arms()
    q0, all_targets = scenario(arms, args.scenario)

    print(f"\n{SCENARIOS[args.scenario]}{'  (step capped at %.2f)' % cap if cap else ''}")
    if view.show_smin:
        print(f"start sigma_min = {arms.sigma_min(q0):.4f}")
    runs = []
    for mk in args.methods:
        m = METHODS[mk]
        targets = select_targets(m, all_targets)
        qs, info = trajectory(arms, m, q0, targets, args.iters, cap, view.show_smin)
        print(f"  {m.label:<46} {caption(info[-1], long=True)}   largest |dq| {max(i['step'] for i in info):.2f} rad")
        # --capped shows controller time: at most one frame per iteration (but plateaus still fast-forward)
        runs.append((m, targets, pace(qs, info, args.speed, max_per_iter=1 if cap else MAX_FRAMES_PER_ITER)))

    sim = MujocoSim(SCENE, JOINT_NAMES)
    sim.set_joint_state(to_names(q0))
    starts = {side: sim.tcp_pose(side)[0] for side in ("left", "right")}
    if view.focus == "left_shoulder":
        lookat = sim.joint_frame("q_l1")[0].copy()   # the arm can swing anywhere within its reach of the shoulder
    else:
        lookat = (sim.joint_frame("q_l1")[0] + sim.joint_frame("q_r1")[0]) / 2 + [0.2, 0.0, -0.1]

    if args.record:
        record(sim, runs, starts, view, lookat, args)
    else:
        run_viewer(sim, runs, starts, view, lookat, args)


if __name__ == "__main__":
    main()
