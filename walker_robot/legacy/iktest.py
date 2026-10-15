import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # walker_robot/, for `from IK import ...`

from IK import ForwardKinematics, DHParameter, InverseKinematics
import sympy as sp
import numpy as np

from walker import Walker
from walkerSimulation import WalkerSimulation
from walkerControl import WalkerControl

import time




def main():
    x,y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7, q_r1, q_r2, q_r3, q_r4, q_r5, q_r6, q_r7 = sp.symbols('x y yaw lift q_l1 q_l2 q_l3 q_l4 q_l5 q_l6 q_l7 q_r1 q_r2 q_r3 q_r4 q_r5 q_r6 q_r7')
    p2 = sp.pi/2
    m_p2 = -1.0 * p2
    h = 0.697 - 0.1 # 0.1 lift sim offset
    l01 = 0.0915
    l12 = 0.061131
    l23 = 0.063
    l34 = 0.157
    l45 = 0.0965
    l56 = 0.1195
    l_eff = 0.114501 + 0.055
    left_dh_params = [
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

    right_dh_params = [
            DHParameter(d=0.0, theta= m_p2, a= 0.0, alpha= m_p2),
            DHParameter(d=x, theta=0.0, a=0.0, alpha=0.0),
            DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2), 
            DHParameter(d=0.0, theta=p2, a=0.0, alpha=m_p2),
            DHParameter(d=y, theta=0.0, a=0.0, alpha=0.0),
            DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
            DHParameter(d=lift, theta=yaw, a=0.0, alpha=0.0),
            DHParameter(d=h, theta=0.0, a=0.0, alpha=0.0),  #BASE
            DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
            DHParameter(d=l01, theta=q_r1 + m_p2, a=0.0, alpha=0.0),#DOF1
            DHParameter(d=l12, theta=0.0, a=0.0, alpha=p2),
            DHParameter(d=0.0, theta=q_r2, a=l23, alpha=0.0), #DOF2
            DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=m_p2),
            DHParameter(d=l34, theta=q_r3 + p2, a=0.0, alpha=0.0), #DOF3
            DHParameter(d=0.0, theta=0.0, a=0.0, alpha=m_p2),
            DHParameter(d=0.0, theta=q_r4 + m_p2, a=0.0, alpha=0.0), #DOF4
            DHParameter(d=0.0, theta=p2, a=0.0, alpha=p2),
            DHParameter(d=l45, theta=q_r5, a=0.0, alpha=0.0), #DOF5
            DHParameter(d=l56, theta=m_p2, a=0.0, alpha=m_p2),
            DHParameter(d=0.0, theta=q_r6 + m_p2, a=0.0, alpha=0.0), #DOF6
            DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
            DHParameter(d=0.0, theta=q_r7, a=l_eff, alpha=0.0), #DOF7
            DHParameter(d=0.0, theta=p2, a=0.0, alpha=p2),  
            DHParameter(d=0.0, theta=m_p2, a=0.0, alpha=0.0)  # End-effector Adjustment to convention
    ]
    
    base_symbols = [x, y, yaw, lift]
    left_arm_symbols = [q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7]
    right_arm_symbols = [q_r1, q_r2, q_r3, q_r4, q_r5, q_r6, q_r7]
    left_symbols = base_symbols + left_arm_symbols      # Jacobian column order for the left FK
    right_symbols = base_symbols + right_arm_symbols    # Jacobian column order for the right FK
    all_symbols = base_symbols + left_arm_symbols + right_arm_symbols
    N_BASE = len(base_symbols)

    left_fk = ForwardKinematics(left_dh_params, variables=left_symbols)
    right_fk = ForwardKinematics(right_dh_params, variables=right_symbols)

    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    sim = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01, showTime=False)
    control = WalkerControl(sim)
    sim.visualize()

    left_solver = InverseKinematics(left_fk)     # create once, reuse
    right_solver = InverseKinematics(right_fk)
    current_joint_state = {s: 0.0 for s in all_symbols}   # one shared state for all 18 joints

    def sub_state(symbols):
        """Per-arm view of the state, in that arm's Jacobian column order."""
        return {s: current_joint_state[s] for s in symbols}

    # --- joint id lookup by name (no hard-coded id assumptions for the right arm) ---
    JOINT_NAME = {
        'x': 'x', 'y': 'y', 'lift': 'lift', 'yaw': 'yaw',
        **{f'q_l{i}': f'openarm_left_joint{i}' for i in range(1, 8)},
        **{f'q_r{i}': f'openarm_right_joint{i}' for i in range(1, 8)},
    }

    def joint_id(symbol):
        return walker.joints_info[JOINT_NAME[symbol.name]]

    def limits_for(symbols):
        lim = np.array([walker.model.jnt_range[joint_id(s)] for s in symbols], dtype=float)
        lim[lim[:, 0] >= lim[:, 1]] = [-np.inf, np.inf]   # MuJoCo [0, 0] -> unlimited
        return lim

    left_limits = limits_for(left_symbols)
    right_limits = limits_for(right_symbols)

    # --- settings ---
    TARGET_INTERVAL = 5.0          # seconds to let the robot travel
    MAX_IK_ITERS = 50
    POS_TOL = 1e-3           # meters, applied to BOTH arms
    LEFT_TARGET_LOW   = np.array([-2.0, 0.0, 0.6])
    LEFT_TARGET_HIGH  = np.array([ 2.0, 2.75, 1.0])
    RIGHT_TARGET_LOW  = np.array([-2.0, -2.75, 0.6])   
    RIGHT_TARGET_HIGH = np.array([ 2.0, 0.0, 1.0])
    rng = np.random.default_rng()

    DAMPING = 0.2
    LIMIT_MARGIN = 0.15
    LIMIT_GAIN = 50.0

    LEFT_DESIRED_POS = {q_l1: 0.0, q_l2: -0.4, q_l3: 0.2, q_l4: 1.7, q_l5: 0.06, q_l6: 0.0, q_l7: 0.0}
    RIGHT_DESIRED_POS = {q_r1: 0.0, q_r2: 0.4, q_r3: -0.2, q_r4: 1.7, q_r5: -0.06, q_r6: 0.0, q_r7: 0.0}

    ik = InverseKinematics()                       # no fk needed
    all_limits = limits_for(all_symbols)           # (18, 2), all_symbols order
    DESIRED_POS = {**LEFT_DESIRED_POS, **RIGHT_DESIRED_POS}
    DESIRED_GAIN = 0.2                             # fraction of the gap to the desired pose closed per iteration

    def solve_to_targets(target_l, target_r):
        """One stacked solve for both arms; desired pose acts only in the null space.
        Returns (iters, err_left, err_right, solve_time_s)."""
        t0 = time.perf_counter()
        chains = [(left_fk, left_symbols, target_l), (right_fk, right_symbols, target_r)]
        err_l = err_r = np.inf
        iters = 0
        for iters in range(1, MAX_IK_ITERS + 1):
            dq = ik.solve_ik_stacked(
                current_joint_state, chains, all_symbols, all_limits,
                damped_factor=DAMPING, margin=LIMIT_MARGIN, gain=LIMIT_GAIN,
                desired_pos=DESIRED_POS, desired_gain=DESIRED_GAIN)
            for k, s in enumerate(all_symbols):
                current_joint_state[s] += dq[k]
            err_l = np.linalg.norm(target_l - left_fk.pos_np(sub_state(left_symbols)))
            err_r = np.linalg.norm(target_r - right_fk.pos_np(sub_state(right_symbols)))
            if err_l < POS_TOL and err_r < POS_TOL:
                break
        return iters, err_l, err_r, time.perf_counter() - t0

    def send_to_sim():
        ids = [joint_id(s) for s in all_symbols]
        vals = [current_joint_state[s] for s in all_symbols]
        control.set_joint_position(ids, vals, vel=5.0, acc=10.0, sync=True)

    def get_actual_eff_pos():
        """Read joint angles from the sim and plug them into both FK models."""
        sim_pos = walker.get_joint_positions()   # {joint_id: qpos}
        actual = {s: sim_pos[joint_id(s)] for s in all_symbols}
        return (left_fk.pos_np({s: actual[s] for s in left_symbols}),
                right_fk.pos_np({s: actual[s] for s in right_symbols}))

    last_model_pos = None   # (left, right) the FK model predicted for the previous command

    while True:
        actual_l, actual_r = get_actual_eff_pos()
        print("\nActual eff positions (sim joints -> FK):  L", np.round(actual_l, 3), " R", np.round(actual_r, 3))
        if last_model_pos is not None:
            print("Gap to commanded model position:  L",
                  np.round(np.linalg.norm(actual_l - last_model_pos[0]), 4),
                  "m  R", np.round(np.linalg.norm(actual_r - last_model_pos[1]), 4), "m")

        target_l = rng.uniform(LEFT_TARGET_LOW, LEFT_TARGET_HIGH)
        # target_r = rng.uniform(RIGHT_TARGET_LOW, RIGHT_TARGET_HIGH)

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)                 # uniform random unit vector
        RIGHT_OFFSET_MAX = 0.75    # max distance of the right target from the left target (m)
        target_r = target_l + direction * rng.uniform(0.0, RIGHT_OFFSET_MAX)

        # yaw desired: face perpendicular to the R->L target line, left side toward the left target.
        # At yaw=0 the base's left axis is +y; at yaw it is (-sin yaw, cos yaw), so align it with d.
        d = (target_l - target_r)[:2]
        if np.linalg.norm(d) > 1e-6:                       # line is undefined if targets stack vertically
            yaw_desired = np.arctan2(-d[0], d[1])
            yaw_now = current_joint_state[yaw]
            yaw_desired = yaw_now + (yaw_desired - yaw_now + np.pi) % (2 * np.pi) - np.pi   # nearest equivalent angle
            DESIRED_POS[yaw] = yaw_desired
            print(f"Yaw desired: {np.degrees(yaw_desired):.1f} deg")

        print(f"New targets: L {np.round(target_l, 3)}  R {np.round(target_r, 3)}")
        sim.set_target_marker(target_l, name="left", rgba=(1.0, 0.0, 0.0, 1.0))
        sim.set_target_marker(target_r, name="right", rgba=(0.0, 0.0, 1.0, 1.0))

        iters, err_l, err_r, solve_time = solve_to_targets(target_l, target_r)
        print(f"IK solved in {solve_time * 1e3:.2f} ms "
              f"({iters} iters, final error L {err_l:.4f} m, R {err_r:.4f} m)")
        last_model_pos = (left_fk.pos_np(sub_state(left_symbols)),
                          right_fk.pos_np(sub_state(right_symbols)))
        print("Reached eff positions (model):  L", np.round(last_model_pos[0], 3),
              " R", np.round(last_model_pos[1], 3))

        send_to_sim()

        t_end = time.perf_counter() + TARGET_INTERVAL
        while time.perf_counter() < t_end:
            control.step()
            time.sleep(0.01)



if __name__ == "__main__":
    main()

