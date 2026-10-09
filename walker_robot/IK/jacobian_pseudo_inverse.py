from .forward_kinematics import ForwardKinematics
import numpy as np


class JacobianPseudoInverse:
    def __init__(self, fk: ForwardKinematics):
        self.fk = fk

    def solve_ik(self, current_joint_state: dict, target: np.array) -> np.array:
        """
        Compute the pseudo-inverse of the given Jacobian matrix using the Moore-Penrose method.
        Args:
            target: [3 x 1] target position as a numpy array.
        Returns:
            changes to the current joint state.
        """
        J = self.fk.jacobian_np(current_joint_state)
        err = target - self.fk.pos_np(current_joint_state)
        return np.linalg.pinv(J) @ err


    def solve_ik_DSL(self, current_joint_state: dict, target: np.array, damped_factor: float) -> np.array:
        """
        Compute the damped pseudo-inverse of the given Jacobian matrix using the Levenberg-Marquardt method.
        Args:
            target: [3 x 1] target position as a numpy array.
            damping_factor: scalar value for damping.
        Returns:
            changes to the current joint state.
        """
        J = self.fk.jacobian_np(current_joint_state)
        err = target - self.fk.pos_np(current_joint_state)

        J_damped = J.T @ np.linalg.inv(J @ J.T + (damped_factor ** 2) * np.eye(J.shape[0]))
        return J_damped @ err

    @staticmethod
    def _limit_weights(q, dq, lim, margin, gain):
        """
        Per-joint cost multiplier w_i >= 1.

        w_i == 1 when joint i is moving away from its nearest limit, or is
        still outside the margin zone. Inside the zone (the last `margin`
        fraction of the range) and moving TOWARD that limit, w_i grows
        smoothly and blows up as q_i reaches the limit.
        Infinite limits (e.g. a base x/y axis) are never penalised.
        """
        lo, hi = lim[:, 0], lim[:, 1]
        rng = hi - lo
        finite = np.isfinite(rng) & (rng > 0)
        safe_rng = np.where(finite, rng, 1.0)

        # normalised distance (0..1) to the limit this joint is heading toward
        dist = np.where(dq >= 0.0, hi - q, q - lo) / safe_rng
        dist = np.where(finite, dist, np.inf)

        p = np.clip((margin - dist) / margin, 0.0, 1.0)  # 0 = far, 1 = at limit
        return 1.0 + gain * p ** 2 / (1.0 - p + 1e-3)

   

    def solve_ik_DSL_limits(
        self,
        current_joint_state: dict,
        target: np.array,
        joint_limits,
        damped_factor=0.3,
        margin: float = 0.15,
        gain: float = 50.0,
        n_refine: int = 2,
        rest_pos=None,
        rest_weight=0.002,
    ) -> np.array:
        """
        Weighted damped least squares with
          (a) joint-limit damping: extra damping when a joint moves toward a limit
          (b) resting-position pull: a soft cost on distance from each joint's rest pos

            minimise ||J dq - err||^2 + dq^T diag(lam^2 * w) dq
                     + sum_i r_i (q_i + dq_i - q_rest_i)^2

        Args:
            current_joint_state: dict {symbol: value}, ordered like the Jacobian columns.
            target: [3] target position.
            joint_limits: (n, 2) array or dict {symbol: (lo, hi)}; +-np.inf = unlimited.
            damped_factor: scalar lambda, or length-n vector.
            margin, gain, n_refine: joint-limit penalty settings (see _limit_weights).
            rest_pos: dict {symbol: rest value} (symbols you leave out have no rest pos),
                or a length-n array with np.nan where a joint has no rest pos. None = off.
            rest_weight: scalar, or length-n vector of per-joint pull strength r_i.
        Returns:
            dq, the change to the current joint state (kept within limits).
        """
        J = self.fk.jacobian_np(current_joint_state)
        err = target - self.fk.pos_np(current_joint_state)
        n = J.shape[1]

        q = np.array([float(v) for v in current_joint_state.values()])
        if isinstance(joint_limits, dict):
            lim = np.array([joint_limits[s] for s in current_joint_state], dtype=float)
        else:
            lim = np.asarray(joint_limits, dtype=float)
        if lim.shape != (n, 2):
            raise ValueError(f"joint_limits must have shape ({n}, 2), got {lim.shape}")

        # resting-position pull: R_i = weight for joints that have a rest pos, else 0
        R = np.zeros(n)
        rest_off = np.zeros(n)  # (q - q_rest), 0 where there is no rest pos
        if rest_pos is not None:
            if isinstance(rest_pos, dict):
                q_rest = np.array([rest_pos.get(s, np.nan) for s in current_joint_state], dtype=float)
            else:
                q_rest = np.asarray(rest_pos, dtype=float)
            if q_rest.shape != (n,):
                raise ValueError(f"rest_pos must have length {n}, got {q_rest.shape}")
            has_rest = np.isfinite(q_rest)
            w_rest = np.broadcast_to(np.asarray(rest_weight, dtype=float), (n,))
            R = np.where(has_rest, w_rest, 0.0)
            rest_off = np.where(has_rest, q - q_rest, 0.0)

        lam2 = np.broadcast_to(np.asarray(damped_factor, dtype=float) ** 2, (n,))
        A0 = J.T @ J + np.diag(R)
        b = J.T @ err - R * rest_off

        def solve(w):
            return np.linalg.solve(A0 + np.diag(lam2 * w), b)

        # limit weights depend on the direction of dq, so refine: solve, reweight, re-solve
        dq = solve(np.ones(n))
        for _ in range(n_refine):
            dq = solve(self._limit_weights(q, dq, lim, margin, gain))

        # safety net: never step past a hard limit
        return np.clip(q + dq, lim[:, 0], lim[:, 1]) - q
    
if __name__ == "__main__":

    import sympy as sp
    from forward_kinematics import ForwardKinematics
    from dh_parameter import DHParameter
    x,y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7 = sp.symbols('x y yaw lift q_l1 q_l2 q_l3 q_l4 q_l5 q_l6 q_l7')
    p2 = sp.pi/2
    m_p2 = -1.0 * p2
    h = 0.697 + 0.1
    l01 = 0.0915
    l12 = 0.061131
    l23 = 0.063
    l34 = 0.157
    l45 = 0.0965
    l56 = 0.1195
    l_eff = 0.114501 + 0.055

    dh_params = [
        DHParameter(d=0.0, theta= m_p2, a= 0.0, alpha= m_p2),
        DHParameter(d=x, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2), 
        DHParameter(d=0.0, theta=p2, a=0.0, alpha=m_p2),
        DHParameter(d=y, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=lift, theta=yaw, a=0.0, alpha=0.0),
        DHParameter(d=h, theta=0.0, a=0.0, alpha=0.0),  #BASE
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha= m_p2),
        DHParameter(d=l01, theta=q_l1 + p2, a=0.0, alpha= 0.0), #DOF1
        DHParameter(d=l12, theta=0.0, a=0.0, alpha= m_p2), #DOF2
        DHParameter(d=0.0, theta=q_l2, a=l23, alpha= 0.0),
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha= m_p2),
        DHParameter(d=l34, theta=q_l3 + p2, a=0.0, alpha= 0.0), #DOF3
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha= m_p2), 
        DHParameter(d=0.0, theta=q_l4 + m_p2, a=0.0, alpha= 0.0), #DOF4
        DHParameter(d=0.0, theta=p2, a=0.0, alpha= p2),
        DHParameter(d=l45, theta=q_l5, a=0.0, alpha= 0.0), #DOF5
        DHParameter(d=l56, theta=m_p2, a=0.0, alpha= m_p2),
        DHParameter(d=0.0, theta=q_l6 + m_p2, a=0.0, alpha= 0.0), #DOF6
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha= m_p2),
        DHParameter(d=0.0, theta=q_l7, a=l_eff, alpha= 0.0), #DOF7
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=m_p2),  
        DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=0.0)  # End-effector Adjustment to convention
    ]
    
    symbols = [x, y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7]

    fk = ForwardKinematics(dh_params, variables=symbols)

    subs = {x: 0.0, y: 0.0, yaw: 0.0, lift: 0.0, q_l1: 0.0, q_l2: 0.0, q_l3: 0.0, q_l4: 0.0, q_l5: 0.0, q_l6: 0.0, q_l7: 0.0}  
    eff = fk.pos_np(subs)

    print("End-Effector Position:")
    print(np.round(eff,3))

    target = np.array([1.0, 1.5, 1.2])  # Example target position
    print("Target Position:")
    print(np.round(target,3))

    current_joint_state = subs.copy()  # Start with the initial joint state

    for i in range(5):  # Perform multiple iterations to converge to the target
        jacobian_pinv_solver = JacobianPseudoInverse(fk)
        joint_changes = jacobian_pinv_solver.solve_ik(current_joint_state, target)


        print("New eff position after applying joint changes:")
        # Update the current joint state with the computed changes
        for i, symbol in enumerate(symbols):
            current_joint_state[symbol] += joint_changes[i] 

        print("Joint Changes:")
        print(np.round(joint_changes,3))
        new_eff = fk.pos_np(current_joint_state)
        print(np.round(new_eff,3))

    print("Final Joint State:")
    for symbol, value in current_joint_state.items():
        print(f"{symbol}: {np.round(value,3)}")