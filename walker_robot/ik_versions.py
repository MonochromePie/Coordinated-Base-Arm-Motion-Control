"""
Every IK version tried so far, as interchangeable one-step functions, plus the loop that iterates them.
Used by the benchmark (benchmark/run_benchmark.py) and the MuJoCo playback (ik_sim/run_ik_sim.py).

Stages 1-4 are the older methods, rebuilt here for the comparison; stage 5 is the current solver
(IK/inverse_kinematics.py, solve_ik_stacked) with the BimanualIK settings from bimanual_ik.py.

    STAGE 1  pinv         Jacobian pseudoinverse     left arm, position only, base locked
                          dq = J^T (J J^T)^-1 e
    STAGE 2  dls          Damped least squares       left arm, position only, base locked
                          dq = (J^T J + lam^2 I)^-1 J^T e
    STAGE 3  rest_left    DLS + rest weight          left arm + base, position + orientation
                          dq = (J^T J + diag(lam^2 w) + b S)^-1 (J^T e + b S (q_rest - q))
                          (w = joint-limit weights, S = diagonal selector of the joints that have a rest value)
    STAGE 4  rest_both    same as stage 3 for both arms stacked, sharing the base joints
    STAGE 5  current      stacked DLS + joint-limit weights, rest pose in the NULL SPACE of J (both arms + base)
             current_arm  stage 5 on the left arm only, position only, base locked (compares with 1 and 2)
"""
import time
from dataclasses import dataclass, field

import numpy as np

from IK import ForwardKinematics, InverseKinematics
from bimanual_ik import (
    BASE_SYMBOLS, LEFT_ARM_SYMBOLS, RIGHT_ARM_SYMBOLS, JOINT_LIMITS, BimanualIK,
    left_dh_params, right_dh_params,
    x, y, yaw, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7, q_r1, q_r2, q_r3, q_r4, q_r5, q_r6, q_r7,
)

# ---- old-solver settings (stage 3/4), as in the earlier rest-weight version (solve_ik_DSL_limits) ----
OLD_DAMPING = 0.3
OLD_REST_POS = {q_l1: 0.0, q_l2: -0.4, q_l3: 0.157, q_l4: 1.58, q_l5: 0.0, q_l6: 0.0, q_l7: 0.0}
# mirror of the left rest pose: q_r2 and q_r3 flip sign (q_r3 = +0.157 would put the right hand
# 9.5 cm further out and 4 cm higher than the left one)
OLD_REST_POS_RIGHT = {q_r1: 0.0, q_r2: 0.4, q_r3: -0.157, q_r4: 1.58, q_r5: 0.0, q_r6: 0.0, q_r7: 0.0}
OLD_REST_WEIGHT = 0.002
LIMIT_MARGIN, LIMIT_GAIN = 0.15, 50.0

# ---- stage 1/2 damping ----
DLS_DAMPING = 0.2           # same as BimanualIK.ARM_DAMPING

LIMITS = {s: JOINT_LIMITS[s.name] for s in BASE_SYMBOLS + LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS}
BASE_ALL = list(BASE_SYMBOLS)


# =====================================================================================
# Kinematic model
# =====================================================================================

class Arms:
    """Left/right FK and a helper that stacks task Jacobians/errors over a set of solved joints."""

    def __init__(self):
        print("Building FK model (symbolic, takes a few seconds)...")
        self.fk = {
            "left": ForwardKinematics(left_dh_params, BASE_SYMBOLS + LEFT_ARM_SYMBOLS),
            "right": ForwardKinematics(right_dh_params, BASE_SYMBOLS + RIGHT_ARM_SYMBOLS),
        }
        self.arm_symbols = {"left": LEFT_ARM_SYMBOLS, "right": RIGHT_ARM_SYMBOLS}
        # lambdify every function once now, so its one-off compile time is not counted as solve time
        q = {s: float(np.clip(0.0, *LIMITS[s])) for s in LIMITS}
        for side, fk in self.fk.items():
            sub = {s: q[s] for s in self.chain_symbols(side)}
            fk.pos_np(sub), fk.jacobian_np(sub), fk.angular_jacobian_np(sub)

    def chain_symbols(self, side):
        return BASE_SYMBOLS + self.arm_symbols[side]

    def pose(self, side, q):
        sub = {s: q[s] for s in self.chain_symbols(side)}
        fk = self.fk[side]
        return fk.pos_np(sub), fk.orientation_matrix_np(sub)

    def task(self, q, targets, solve, orientation_weight=0.5):
        """
        Stacked J (rows: 3 position [+3 weighted orientation] per arm, cols: `solve`) and error e.
        targets: {side: (pos, R or None)}.
        """
        col = {s: i for i, s in enumerate(solve)}
        Js, es = [], []
        for side, (tp, tR) in targets.items():
            syms = self.chain_symbols(side)
            sub = {s: q[s] for s in syms}
            fk = self.fk[side]
            Jc = fk.jacobian_np(sub)
            ec = tp - fk.pos_np(sub)
            if tR is not None:
                Jc = np.vstack([Jc, orientation_weight * fk.angular_jacobian_np(sub)])
                ec = np.concatenate([ec, orientation_weight * fk.rotation_error(tR, fk.orientation_matrix_np(sub))])
            Jf = np.zeros((Jc.shape[0], len(solve)))
            for k, s in enumerate(syms):
                if s in col:
                    Jf[:, col[s]] = Jc[:, k]
            Js.append(Jf)
            es.append(ec)
        return np.vstack(Js), np.concatenate(es)

    def errors(self, q, targets):
        """Worst position error [m] and worst orientation error [rad] (0 if none) over the arms."""
        pe, re = 0.0, 0.0
        for side, (tp, tR) in targets.items():
            p, R = self.pose(side, q)
            pe = max(pe, float(np.linalg.norm(tp - p)))
            if tR is not None:
                re = max(re, float(np.linalg.norm(ForwardKinematics.rotation_error(tR, R))))
        return pe, re

    def side_errors(self, q, targets):
        """{side: position error [m]} for every arm with a target."""
        return {side: float(np.linalg.norm(tp - self.pose(side, q)[0])) for side, (tp, _) in targets.items()}

    def sigma_min(self, q):
        """Smallest singular value of the left arm's position Jacobian (arm joints): 0 = singular."""
        J, _ = self.task(q, {"left": (np.zeros(3), None)}, LEFT_ARM_SYMBOLS)
        return float(np.linalg.svd(J, compute_uv=False)[-1])


# =====================================================================================
# The methods. Each returns dq (in `solve` order) for the current q.
# =====================================================================================

def vec(q, solve):
    return np.array([q[s] for s in solve], dtype=float)


def limits_of(solve):
    return np.array([LIMITS[s] for s in solve], dtype=float)


def step_pinv(arms, q, targets, solve):
    """Stage 1: textbook pseudoinverse dq = J^T (J J^T)^-1 e. Blows up as J J^T becomes singular."""
    J, e = arms.task(q, targets, solve)
    return J.T @ np.linalg.solve(J @ J.T, e)


def step_dls(arms, q, targets, solve):
    """Stage 2: damped least squares dq = (J^T J + lam^2 I)^-1 J^T e."""
    J, e = arms.task(q, targets, solve)
    return np.linalg.solve(J.T @ J + DLS_DAMPING ** 2 * np.eye(len(solve)), J.T @ e)


def step_rest(arms, q, targets, solve, rest_pos):
    """
    Stages 3/4: limit-aware DLS with the rest pose as an extra weighted cost
        min ||J dq - e||^2 + dq^T diag(lam^2 w) dq + b ||S (q + dq - q_rest)||^2
    At a fixed point (dq = 0): J^T e = -b S (q_rest - q), so e != 0 unless the solution happens to
    sit at the rest pose. The rest pull and the task trade off, which is the steady-state error.
    """
    J, e = arms.task(q, targets, solve)
    v, lim, n = vec(q, solve), limits_of(solve), len(solve)
    sel = np.array([s in rest_pos for s in solve], dtype=float)
    rest = np.array([rest_pos.get(s, 0.0) for s in solve], dtype=float)
    A0 = J.T @ J + OLD_REST_WEIGHT * np.diag(sel)
    b = J.T @ e + OLD_REST_WEIGHT * sel * (rest - v)
    lam2 = OLD_DAMPING ** 2
    dq = np.linalg.solve(A0 + lam2 * np.eye(n), b)
    for _ in range(2):
        w = InverseKinematics._limit_weights(v, dq, lim, LIMIT_MARGIN, LIMIT_GAIN)
        dq = np.linalg.solve(A0 + np.diag(lam2 * w), b)
    return dq


_SOLVER = InverseKinematics()


def step_current(arms, q, targets, solve):
    """Stage 5: the current solver with BimanualIK's settings (without its MAX_JOINT_STEP cap)."""
    chains = []
    for side, (tp, tR) in targets.items():
        chains.append((arms.fk[side], arms.chain_symbols(side), tp, tR))
    damping = [BimanualIK.BASE_DAMPING if s in BASE_SYMBOLS else BimanualIK.ARM_DAMPING for s in solve]
    return _SOLVER.solve_ik_stacked(
        current_joint_state=q, chains=chains, all_symbols=solve,
        joint_limits={s: LIMITS[s] for s in solve},
        damped_factor=damping,
        desired_pos={x: 0.0, y: 0.0, yaw: 0.0},
        desired_gain=0.05,
        orientation_weight=BimanualIK.ORIENTATION_WEIGHT,
    )


@dataclass
class Method:
    key: str
    label: str
    step: callable
    sides: tuple            # arms with a target
    solve: list             # joints being solved for
    orientation: bool       # 6-D targets (else position only)
    color: str


_LEFT_BASE = BASE_ALL + LEFT_ARM_SYMBOLS
_BOTH_BASE = BASE_ALL + LEFT_ARM_SYMBOLS + RIGHT_ARM_SYMBOLS

METHODS = {m.key: m for m in [
    Method("pinv", "1. Pseudoinverse (L, pos)", step_pinv,
           ("left",), LEFT_ARM_SYMBOLS, False, "#eb6834"),
    Method("dls", "2. DLS (L, pos)", step_dls,
           ("left",), LEFT_ARM_SYMBOLS, False, "#1baf7a"),
    Method("rest_left", "3. DLS + rest weight (L+base, pose)",
           lambda a, q, t, s: step_rest(a, q, t, s, OLD_REST_POS),
           ("left",), _LEFT_BASE, True, "#2a78d6"),
    Method("rest_both", "4. DLS + rest weight (L+R+base, pose)",
           lambda a, q, t, s: step_rest(a, q, t, s, OLD_REST_POS | OLD_REST_POS_RIGHT),
           ("left", "right"), _BOTH_BASE, True, "#6250d6"),
    Method("current", "5. Current: null-space rest (L+R+base, pose)", step_current,
           ("left", "right"), _BOTH_BASE, True, "#e87ba4"),
    Method("current_arm", "5. Current, arm only (L, pos)", step_current,
           ("left",), LEFT_ARM_SYMBOLS, False, "#c4900f"),
]}


def select_targets(method: Method, targets: dict) -> dict:
    """The part of `targets` ({side: (pos, R)}) a method works on: its arms, rotation only if it tracks it."""
    return {side: (tp, tR if method.orientation else None) for side, (tp, tR) in targets.items()
            if side in method.sides}


# =====================================================================================
# Iteration loop
# =====================================================================================

@dataclass
class Run:
    converged: bool
    iters: int
    pos_err: float
    rot_err: float
    time_s: float                        # total wall time (FK + Jacobian + solve), all iterations
    max_step: float                      # largest |dq_i| seen in any iteration (before clipping/capping)
    q_final: dict
    pos_hist: list = field(default_factory=list)
    rot_hist: list = field(default_factory=list)
    step_hist: list = field(default_factory=list)
    qs: list = field(default_factory=list)   # joint state after every iteration, qs[0] = start
    failed: str = ""                     # e.g. "singular matrix"


def run(arms, method: Method, q0: dict, targets: dict, pos_tol, rot_tol, max_iters, cap=None, record=False):
    """
    Iterate q <- clip(q + dq, limits) from q0 until the error is under the tolerance or max_iters is
    reached. cap: optional largest joint step per iteration (BimanualIK.MAX_JOINT_STEP = controller
    behaviour); without it the numbers describe the solver itself. record: keep the per-iteration history.
    """
    q = dict(q0)
    lim = limits_of(method.solve)
    pe, re = arms.errors(q, targets)
    pos_hist, rot_hist, step_hist, qs = ([pe], [re], [], [dict(q)]) if record else ([], [], [], [])
    max_step, failed, it = 0.0, "", 0
    t0 = time.perf_counter()
    for it in range(1, max_iters + 1):
        if pe < pos_tol and re < rot_tol:
            it -= 1
            break
        try:
            dq = method.step(arms, q, targets, method.solve)
        except np.linalg.LinAlgError:
            failed = "singular matrix"
            break
        if not np.all(np.isfinite(dq)):
            failed = "non-finite step"
            break
        biggest = float(np.max(np.abs(dq)))
        max_step = max(max_step, biggest)
        if cap and biggest > cap:
            dq = dq * cap / biggest
        q = dict(q)
        q.update(zip(method.solve, np.clip(vec(q, method.solve) + dq, lim[:, 0], lim[:, 1])))
        pe, re = arms.errors(q, targets)
        if record:
            pos_hist.append(pe)
            rot_hist.append(re)
            step_hist.append(biggest)
            qs.append(q)
    elapsed = time.perf_counter() - t0
    converged = not failed and pe < pos_tol and re < rot_tol
    return Run(converged, it, pe, re, elapsed, max_step, q, pos_hist, rot_hist, step_hist, qs, failed)
