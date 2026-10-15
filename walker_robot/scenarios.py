"""
Test scenarios shared by the benchmark (benchmark/run_benchmark.py) and the MuJoCo playback
(ik_sim/run_ik_sim.py): start poses and end-effector targets.

    A1  singularity, reachable: from a nearly straight left arm (elbow q_l4 = 0.05 rad), raise the hand 10 cm
    A2  singularity, out of reach: same start, push the hand 10 cm further down (the arm stretches into it)
    B   rest weight: both arms start at the rest pose; both hands get an asymmetric 6-D target that needs
        the base and lift
    random_config: random reachable configurations for the benchmark's experiment C

Targets are {side: (pos [3], R [3x3] or None)}; ik_versions.select_targets trims them to what a method tracks.
"""
import numpy as np

from bimanual_ik import LEFT_ARM_SYMBOLS, RIGHT_ARM_SYMBOLS, x, y, yaw, lift, q_l1, q_l2, q_l4, q_l7, \
    q_r1, q_r2, q_r3, q_r4, q_r6
from ik_versions import LIMITS, OLD_REST_POS, OLD_REST_POS_RIGHT

# start pose for experiments B and C: elbows bent (the all-zero home pose is exactly singular)
READY = {s: float(np.clip(0.0, *LIMITS[s])) for s in LIMITS}
READY.update({lift: 0.2, q_l4: 1.2, q_r4: 1.2})
# start pose for scenario B: both arms at the stage 3/4 rest pose
REST_START = {**READY, **OLD_REST_POS, **OLD_REST_POS_RIGHT}

SCENARIOS = {
    "A1": "A1 raise hand 10 cm (reachable)",
    "A2": "A2 push hand 10 cm down (out of reach)",
    "B": "B rest weight: one arm vs two arms vs null-space rest",
}
SINGULARITY_OFFSETS = {"A1": np.array([0.0, 0.0, 0.10]), "A2": np.array([0.0, 0.0, -0.10])}


def singularity_start():
    """Left arm nearly straight, hanging down (elbow q_l4 = 0.05 rad); right arm straight."""
    q0 = dict(READY)
    q0.update({lift: 0.2, q_l4: 0.05, q_r4: 0.0})
    return q0


def rest_target_state():
    """
    Scenario B's target configuration (the hand poses are its FK). The two arms are posed differently
    (left reaching high and bent, right low and to the side), so the shared base/lift joints cannot sit
    where both arms AND the rest pose want them.
    """
    qt = dict(READY)
    qt.update({x: 0.1, yaw: 0.3, lift: 0.4,
               q_l1: -1.2, q_l2: -0.2, q_l4: 2.0, q_l7: 0.6,
               q_r1: 0.3, q_r2: 1.0, q_r3: 0.5, q_r4: 1.0, q_r6: 0.3})
    return qt


def pose_targets(arms, q):
    """6-D targets for both hands = their poses at configuration q."""
    return {side: arms.pose(side, q) for side in ("left", "right")}


def scenario(arms, key):
    """(start state, full targets) of scenario A1, A2 or B."""
    if key in SINGULARITY_OFFSETS:
        q0 = singularity_start()
        p_left, _ = arms.pose("left", q0)
        # left: position only, start + offset. Right: held at its start pose (for methods that move it)
        return q0, {"left": (p_left + SINGULARITY_OFFSETS[key], None), "right": arms.pose("right", q0)}
    if key == "B":
        return REST_START, pose_targets(arms, rest_target_state())
    raise KeyError(key)


def random_config(rng):
    """Random joint configuration, 10% inside each limit; base kept near home."""
    q = dict(READY)
    for s in LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS:
        lo, hi = LIMITS[s]
        pad = 0.1 * (hi - lo)
        q[s] = rng.uniform(lo + pad, hi - pad)
    q[x], q[y] = rng.uniform(-0.2, 0.2, size=2)
    q[yaw] = rng.uniform(-0.3, 0.3)
    q[lift] = rng.uniform(0.1, 0.6)
    return q
