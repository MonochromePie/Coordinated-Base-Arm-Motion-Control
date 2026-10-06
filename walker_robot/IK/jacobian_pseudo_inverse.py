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