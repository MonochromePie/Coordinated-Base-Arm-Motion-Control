from dh_parameter import DHParameter
import sympy as sp

class ForwardKinematics:
    
    def __init__(self, dh_parameters: list[DHParameter], variables: list[sp.Symbol] = None):
        self.dh_parameters = dh_parameters
        self.variables = variables if variables is not None else []
        self.transformation_matrix: sp.Matrix = self.get_transformation_matrix()
        self.eff_pos = self.transformation_matrix[:3, 3] 
        self.eff_orientation_matrix = self.transformation_matrix[:3, :3]  

        self._orientation_ZYX = self.get_euler_angles_ZYX()
        self.jacobian = self.eff_pos.jacobian(self.variables) if self.variables else None

        self._pose = sp.Matrix([self.eff_pos, self._orientation_ZYX])
        self.pose_jacobian = self._pose.jacobian(self.variables) if self.variables else None
        
    def get_transformation_matrix(self) -> sp.Matrix:
        """
        Calculate the overall transformation matrix from the base frame to the end-effector frame.
        Returns:
            A 4x4 sympy Matrix representing the overall transformation matrix.
        """
        T_total = sp.eye(4)  

        for dh_param in self.dh_parameters:
            T = dh_param.get_transformation_matrix()
            T_total = T_total * T 

        return sp.Matrix(T_total)

    def get_euler_angles_ZYX(self) -> sp.Matrix:
       
        R = self.eff_orientation_matrix
        roll = sp.atan2(R[2, 1], R[2, 2])
        pitch = sp.atan2(-R[2, 0], sp.sqrt(R[0,0]**2 + R[1,0]**2))
        yaw = sp.atan2(R[1, 0], R[0, 0])

        return sp.Matrix([[roll], [pitch], [yaw]])  

    def get_euler_angles_XYZ(self) -> sp.Matrix:

        R = self.eff_orientation_matrix
        roll = sp.atan2(-R[1, 2], R[2, 2])
        pitch = sp.atan2(R[0, 2], sp.sqrt(R[0,0]**2 + R[0,1]**2))
        yaw = sp.atan2(-R[0, 1], R[0, 0])

        return sp.Matrix([[roll], [pitch], [yaw]])

    def get_quaternion(self) -> sp.Matrix:
        """
        Convert the end-effector rotation matrix to a unit quaternion [qw, qx, qy, qz]
        using Shepperd's method (stable for all rotations, including 180° ones).
        The branch is a sp.Piecewise, so it resolves once numeric values are substituted.
        """
        R = self.eff_orientation_matrix
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        # Branch 1: trace is the largest -> qw dominant
        s1 = 2 * sp.sqrt(1 + trace)
        q1 = [s1 / 4,
              (R[2, 1] - R[1, 2]) / s1,
              (R[0, 2] - R[2, 0]) / s1,
              (R[1, 0] - R[0, 1]) / s1]

        # Branch 2: R00 is the largest -> qx dominant
        s2 = 2 * sp.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q2 = [(R[2, 1] - R[1, 2]) / s2,
              s2 / 4,
              (R[0, 1] + R[1, 0]) / s2,
              (R[0, 2] + R[2, 0]) / s2]

        # Branch 3: R11 is the largest -> qy dominant
        s3 = 2 * sp.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q3 = [(R[0, 2] - R[2, 0]) / s3,
              (R[0, 1] + R[1, 0]) / s3,
              s3 / 4,
              (R[1, 2] + R[2, 1]) / s3]

        # Branch 4: R22 is the largest -> qz dominant
        s4 = 2 * sp.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q4 = [(R[1, 0] - R[0, 1]) / s4,
              (R[0, 2] + R[2, 0]) / s4,
              (R[1, 2] + R[2, 1]) / s4,
              s4 / 4]

        cond1 = trace >= R[0, 0]
        cond1 = sp.And(trace >= R[0, 0], trace >= R[1, 1], trace >= R[2, 2])
        cond2 = sp.And(R[0, 0] >= R[1, 1], R[0, 0] >= R[2, 2])
        cond3 = R[1, 1] >= R[2, 2]

        comps = []
        for i in range(4):
            comps.append(sp.Piecewise(
                (q1[i], cond1),
                (q2[i], cond2),
                (q3[i], cond3),
                (q4[i], True),
            ))

        return sp.Matrix(comps)  # [qw, qx, qy, qz]

    def quaternions_to_euler_XYZ(self, quarternions: sp.Matrix) -> sp.Matrix:
        """
        Convert a unit quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw) in XYZ convention.
        """
        qw, qx, qy, qz = quarternions

        # Roll (X-axis rotation)
        roll = sp.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

        # Pitch (Y-axis rotation)
        pitch = sp.asin(2 * (qw * qy - qz * qx))

        # Yaw (Z-axis rotation)
        yaw = sp.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        return sp.Matrix([[roll], [pitch], [yaw]])
    
if __name__ == "__main__":
    # Example usage with symbolic variables
    x,y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7 = sp.symbols('x y yaw lift q_l1 q_l2 q_l3 q_l4 q_l5 q_l6 q_l7')
    p2 = sp.pi/2
    m_p2 = -1.0 * p2
    h = 0.697
    l01 = 0.0915
    l12 = 0.061131
    l23 = 0.063
    l34 = 0.157
    l45 = 0.0965
    l56 = 0.1195
    l_eff = 0.114501 + 0.055

    symbols = [x, y, yaw, lift, q_l1, q_l2, q_l3, q_l4, q_l5, q_l6, q_l7]
    dh_params = [
        DHParameter(d=0.0, theta= m_p2, a= 0.0, alpha= m_p2),
        DHParameter(d=x, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2), 
        DHParameter(d=0.0, theta=p2, a=0.0, alpha=m_p2),
        DHParameter(d=y, theta=0.0, a=0.0, alpha=0.0),
        DHParameter(d=0.0, theta=0.0, a=0.0, alpha=p2),
        DHParameter(d=0.0, theta=yaw, a=0.0, alpha=0.0),
        DHParameter(d=lift, theta=0.0, a=0.0, alpha=0.0),
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
    
    fk = ForwardKinematics(dh_params, variables=symbols)
    transformation_matrix = fk.get_transformation_matrix()
 
    subs = {x: 0.0, y: 0.0, yaw: 0.0, lift: 0.0, q_l1: 0.0, q_l2: -sp.pi/2, q_l3: 0.0, q_l4: 0.0, q_l5: 0.0, q_l6: 0.0, q_l7: 0.0}  # Substitute some values for demonstration
    print("End-Effector Position:")
    print(fk.eff_pos.subs(subs))  # Substitute some values for demonstration

    print("End-Effector Orientation (Quaternion):")
    print(fk.get_quaternion().subs(subs))  # Substitute some values for demonstration

    print("End-Effector Orientation (Euler Angles ZYX):")
    print(fk.get_euler_angles_ZYX().subs(subs))  # Substitute some

    # print("Jacobian Matrix:")
    # print(fk.jacobian.subs(subs))

    # print("Pose Jacobian Matrix:")
    # print(fk.pose_jacobian.subs(subs))
