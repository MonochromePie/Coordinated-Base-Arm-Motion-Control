from IK import ForwardKinematics, DHParameter, JacobianPseudoInverse
import sympy as sp
import numpy as np

from walker import Walker
from walkerSimulation import WalkerSimulation
from walkerControl import WalkerControl

import time


def symbol_name_to_index(symbol_name):
    symbol_mapping = {
        'x': 0,
        'y': 1,
        'lift': 2,
        'yaw': 3,
        'q_l1': 4,
        'q_l2': 5,
        'q_l3': 6,
        'q_l4': 7,
        'q_l5': 8,
        'q_l6': 9,
        'q_l7': 10
    }
    return symbol_mapping.get(symbol_name, None)


def main():
    
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

    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    sim = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01, showTime=False)
    control = WalkerControl(sim)
    sim.visualize()

    jacobian_pinv_solver = JacobianPseudoInverse(fk)   # create once, reuse
    current_joint_state = {symbol: 0.0 for symbol in symbols}

    # --- settings ---
    TARGET_INTERVAL = 5.0          # seconds to let the robot travel
    MAX_IK_ITERS = 20
    POS_TOL = 1e-3                 # meters
    TARGET_LOW  = np.array([-1.5, -1.5, 0.1])   # x, y, z bounds for random targets
    TARGET_HIGH = np.array([ 1.5,  1.5, 1.5])
    rng = np.random.default_rng()


    def solve_to_target(target):
        """Iterate the pseudo-inverse IK from the current state. Returns (iters, final_error, solve_time_s)."""
        t0 = time.perf_counter()
        err = np.inf
        iters = 0
        for iters in range(1, MAX_IK_ITERS + 1):
            joint_changes = jacobian_pinv_solver.solve_ik(current_joint_state, target)
            for k, symbol in enumerate(symbols):
                current_joint_state[symbol] += joint_changes[k]
            err = np.linalg.norm(target - fk.pos_np(current_joint_state))
            if err < POS_TOL:
                break
        return iters, err, time.perf_counter() - t0


    def send_to_sim():
        pos_dict = {symbol_name_to_index(s.name): current_joint_state[s] for s in symbols}
        control.set_joint_position(list(pos_dict.keys()), list(pos_dict.values()),
                                vel=5.0, acc=10.0, sync=True)


    def get_actual_eff_pos():
        """Read joint angles (ids 0-10) from the sim and plug them into the FK model."""
        sim_pos = walker.get_joint_positions()   # {joint_id: qpos}
        actual_state = {s: sim_pos[symbol_name_to_index(s.name)] for s in symbols}
        return fk.pos_np(actual_state)


    last_model_pos = None   # what the FK model predicted for the previous command

    while True:
        # --- actual eff position from sim joints, before choosing a new target ---
        actual_pos = get_actual_eff_pos()
        print("\nActual eff position (sim joints -> FK):", np.round(actual_pos, 3))
        if last_model_pos is not None:
            print("Gap to commanded model position:",
                    np.round(np.linalg.norm(actual_pos - last_model_pos), 4), "m")

        target = rng.uniform(TARGET_LOW, TARGET_HIGH)
        print(f"New target: {np.round(target, 3)}")

        iters, err, solve_time = solve_to_target(target)
        print(f"IK solved in {solve_time * 1e3:.2f} ms "
                f"({iters} iters, final error {err:.4f} m)")
        last_model_pos = fk.pos_np(current_joint_state)
        print("Reached eff position (sim):", np.round(last_model_pos, 3))

        send_to_sim()

        t_end = time.perf_counter() + TARGET_INTERVAL
        while time.perf_counter() < t_end:
            control.step()
            time.sleep(0.01)


if __name__ == "__main__":
    main()