from pathlib import Path
from typing import Mapping, Sequence
import mujoco

#Im rewritting this fuck ass AI generated code later, but it works for now as a testing ground.
class WalkerControl:
    """Position-control interface for the robot defined in simulation/walker.xml."""

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

    def __init__(self, xml_path: str | Path, sim_dt: float = 0.002, vis_dt: float = 0.1) -> None:
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        self.sim_dt = sim_dt
        self.vis_dt = vis_dt

        self._joint_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in self.BASE_JOINTS
            + self.LEFT_ARM_JOINTS
            + self.RIGHT_ARM_JOINTS
            + self.LEFT_FINGER_JOINTS
            + self.RIGHT_FINGER_JOINTS
        }

        self._actuator_ids = {
            self.model.actuator(i).name: i
            for i in range(self.model.nu)
        }

        self._arm_position_targets: dict[str, float] = {}

        self.reset()

    def reset(self, qpos: Sequence[float] | None = None) -> None:
        """Reset the simulation and initialize position targets."""
        mujoco.mj_resetData(self.model, self.data)

        if qpos is not None:
            if len(qpos) != self.model.nq:
                raise ValueError(
                    f"Expected {self.model.nq} qpos values, got {len(qpos)}."
                )
            self.data.qpos[:] = qpos

        mujoco.mj_forward(self.model, self.data)

        self._arm_position_targets = {
            joint_name: float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            for joint_name, joint_id in self._joint_ids.items()
            if joint_name in self.LEFT_ARM_JOINTS + self.RIGHT_ARM_JOINTS
        }

        # Initialize every position actuator at its current joint position.
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            qpos_address = self.model.jnt_qposadr[joint_id]
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_address]

        self._clip_controls()

    def step(self) -> None:
        """Advance the simulation, according to the visualization time step."""
        for i in range(int(self.vis_dt / self.model.opt.timestep)):
          mujoco.mj_step(self.model, self.data)

    def set_joint_targets(self, targets: Mapping[str, float]) -> None:
        """Set position targets using joint names."""
        for joint_name, target in targets.items():
            if joint_name not in self._joint_ids:
                raise KeyError(f"Unknown joint: {joint_name}")

            joint_id = self._joint_ids[joint_name]
            actuator_id = self._actuator_for_joint(joint_id)

            if actuator_id is None:
                raise ValueError(f"No actuator found for joint: {joint_name}")

            self.data.ctrl[actuator_id] = float(target)

        self._clip_controls()

    def set_arm(self, sides: str, mode: str, targets: Sequence[float]) -> None:

        if sides == "left":
            joint_names = self.LEFT_ARM_JOINTS
        elif sides == "right":
            joint_names = self.RIGHT_ARM_JOINTS
        else:
            raise ValueError(f"Invalid side: {sides}")

        if len(targets) != len(joint_names):
            raise ValueError(
                f"Expected {len(joint_names)} {mode} targets, got {len(targets)}."
            )

        if mode == "pos":
            position_targets = {
                joint_name: float(reference)
                for joint_name, reference in zip(joint_names, targets)
            }
            
        elif mode == "vel":
            position_targets = {
                joint_name: self._arm_position_targets[joint_name]
                + float(reference_velocity) * self.vis_dt
                for joint_name, reference_velocity in zip(joint_names, targets)
            }
        
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.set_joint_targets(position_targets)

        for joint_name in joint_names:
            actuator_id = self._actuator_for_joint(self._joint_ids[joint_name])
            if actuator_id is not None:
                self._arm_position_targets[joint_name] = float(
                    self.data.ctrl[actuator_id]
                )


    def set_gripper(self, sides: str = "both", opening: float = 0.0) -> None:
        """0.0 = closed, 1.0 = fully open"""
        if not (0.0 <= opening <= 1.0):
            raise ValueError(f"Opening must be between 0.0 and 1.0, got {opening}")

        _MAX_OPENING = 0.044
        _MIN_OPENING = 0.0

        opening = _MIN_OPENING + opening * (_MAX_OPENING - _MIN_OPENING)

        if sides == "both":
            self.set_joint_targets({
                joint_name: opening
                for joint_name in self.LEFT_FINGER_JOINTS + self.RIGHT_FINGER_JOINTS
            })
        elif sides == "left":
            self.set_joint_targets({
                joint_name: opening
                for joint_name in self.LEFT_FINGER_JOINTS
            })
        elif sides == "right":
            self.set_joint_targets({
                joint_name: opening
                for joint_name in self.RIGHT_FINGER_JOINTS
            })


    def set_base(
        self,
        x: float | Sequence[float] | None = None,
        y: float | None = None,
        lift: float | None = None,
        yaw: float | None = None,
    ) -> None:
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            if any(value is not None for value in (y, lift, yaw)):
                raise ValueError(
                    "Do not combine a base target sequence with scalar targets."
                )
            self._set_sequence(self.BASE_JOINTS, x)
            return

        self._set_sequence(self.BASE_JOINTS, [x, y, lift, yaw])

    def get_joint_positions(self) -> dict[str, float]:
        """Return current joint positions."""
        positions = {}

        for name, joint_id in self._joint_ids.items():
            qpos_address = self.model.jnt_qposadr[joint_id]
            positions[name] = float(self.data.qpos[qpos_address])

        return positions

    def get_site_position(self, site_name: str) -> tuple[float, float, float]:
        """Return a site's world-frame position."""
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )

        if site_id < 0:
            raise KeyError(f"Unknown site: {site_name}")

        return tuple(float(value) for value in self.data.site_xpos[site_id])

    def _set_sequence(
        self,
        joint_names: Sequence[str],
        targets: Sequence[float | None],
    ) -> None:
        if len(targets) != len(joint_names):
            raise ValueError(
                f"Expected {len(joint_names)} targets, got {len(targets)}."
            )

        self.set_joint_targets({
            joint_name: target
            for joint_name, target in zip(joint_names, targets)
            if target is not None
        })

    def _actuator_for_joint(self, joint_id: int) -> int | None:
        for actuator_id in range(self.model.nu):
            if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                return actuator_id
        return None

    def _clip_controls(self) -> None:
        for actuator_id in range(self.model.nu):
            if self.model.actuator_ctrllimited[actuator_id]:
                lower, upper = self.model.actuator_ctrlrange[actuator_id]
                self.data.ctrl[actuator_id] = max(
                    lower,
                    min(upper, self.data.ctrl[actuator_id]),
                )


if __name__ == "__main__":
    from mujoco import viewer 
    import time

    walker = WalkerControl("simulation/walker_scene.xml", sim_dt=0.002, vis_dt=0.1)
    model = walker.model
    data = walker.data

    with viewer.launch_passive(model, data) as sim:

        sim.sync()
        start_time = time.time()

        while sim.is_running():


            #example of setting the base, arms and grippers to a specific position
            # walker.set_base([0.0, 0.0, 0.0, 0.2])
            last_speed = 0.0
            current_speed = 0.0
            if time.time() - start_time < 2:
                walker.set_arm(sides="left", mode="pos", targets=[-0.6, -0.6, 1.57, 1.7, 0.0, 0.0, 0.0])
                walker.set_arm(sides="right", mode="pos", targets=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            elif time.time() - start_time < 4:
                walker.set_arm(sides="left", mode="pos", targets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                walker.set_arm(sides="right", mode="pos", targets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            else:
                walker.set_arm(sides="left", mode="vel", targets=[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                current_speed = data.qpos[walker._joint_ids["openarm_left_joint1"]]
                print("sim_speed_joint_1: ", (last_speed - current_speed) / walker.vis_dt)
                last_speed = current_speed

            # walker.set_arm(sides="right", mode="pos", targets=[1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print("sim time passed: ", data.time)


            walker.step()
            sim.sync()

            

    


            
            
