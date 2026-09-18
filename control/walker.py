from dataclasses import dataclass

import mujoco
from pathlib import Path


@dataclass
class WalkerInfo:
    xml_path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    joints_info: dict[str, int] 
    """joint name and joint id."""
    joints_pos: dict[input, float] = None
    """joint id and current position."""
    

class Walker:

    LEFT_ARM_JOINTS = [
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
    ]

    RIGHT_ARM_JOINTS = [
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
    ]

    LEFT_FINGER_JOINTS = [
        "openarm_left_finger_joint1", 
        "openarm_left_finger_joint2"
    ]

    RIGHT_FINGER_JOINTS = [
        "openarm_right_finger_joint1",
        "openarm_right_finger_joint2"
    ]

    BASE_JOINTS = ["x", "y", "lift", "yaw"]

    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = self.BASE_JOINTS + self.LEFT_ARM_JOINTS + self.RIGHT_ARM_JOINTS + self.LEFT_FINGER_JOINTS + self.RIGHT_FINGER_JOINTS

        self.joints_info = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in self.joint_names
        }

        self.joints_pos = self.get_joint_positions()
        self.vis_dt = None   

    def get_joint_positions(self) -> dict[int, float]:
        """Return current joint positions."""
        positions = {}

        for name, joint_id in self.joints_info.items():
            qpos_address = self.model.jnt_qposadr[joint_id]
            positions[joint_id] = float(self.data.qpos[qpos_address])

        return positions

    def get_walker_info(self) -> WalkerInfo:
        return WalkerInfo(
            xml_path=self.xml_path,
            model=self.model,
            data=self.data,
            joints_info=self.joints_info,
            joints_pos=self.get_joint_positions()
        )

    

