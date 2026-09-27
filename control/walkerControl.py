from walkerSimulation import WalkerSimulation
from walker import Walker, WalkerInfo
from pid import PID
import mujoco
import threading
import time

class WalkerControl:
    def __init__(self, walkerSimulation: WalkerSimulation):
        self.simulation = walkerSimulation
        self.walker: Walker = self.simulation.walker
        self.walkerInfo: WalkerInfo = self.walker.get_walker_info()
        self.data: mujoco.MjData = self.walkerInfo.data
        self.model: mujoco.MjModel = self.walkerInfo.model

        self._targeted_joint_positions: dict[int, float] = self.walker.get_joint_positions()
        self._last_control_time = self.data.time
        self._control_dt = 0.0

        self.PIDs: dict[int, PID] = {joint_id: PID(kp=1.0, ki=0.05, kd=0.1, max_integral=1.0, max_output=1.0) 
                    for joint_id in self._targeted_joint_positions.keys()}


    def set_joint(self, joint_ids: list[int] , pos: list[float]) -> None:
        if len(joint_ids) != len(pos):
            raise ValueError("Length of joint_ids and pos must be the same.")

        for joint_id, target_pos in zip(joint_ids, pos):
            self._targeted_joint_positions[joint_id] = target_pos

    def _control_joint(self) -> None:

        if self.data.time - self._last_control_time < 0.001:
            pass  # Skip control update if not enough time has passed
        else:  
            current_qpos = self.walker.get_joint_positions()
            
            for i, joint_id in enumerate(self._targeted_joint_positions.keys()):
                        actuator_id = self.joint2actuatorID(joint_id)
                        if actuator_id is not None:
                            error = self._targeted_joint_positions[joint_id] - current_qpos[joint_id]

                            pid_output = self.PIDs[joint_id].update(error, self._control_dt)
                            clamped_pos = self._clamp_control(joint_id, self._targeted_joint_positions[joint_id] + pid_output)
            
                            self.data.ctrl[actuator_id] = clamped_pos

                            self._control_dt = self.data.time - self._last_control_time
                            self._last_control_time = self.data.time 

    def joint2actuatorID(self, joint_id: int) -> int | None:
            for actuator_id in range(self.model.nu):
                if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                    return actuator_id
            return None
    
    def set_joint_velocity(self, joint_ids: list[int], joint_vel: list[float]) -> None:
        for joint_id, vel in zip(joint_ids, joint_vel):
            added_pos = vel * self._control_dt
            self._targeted_joint_positions[joint_id] += added_pos
        

    def jointID2name(self, joint_ids: list[int]) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_ids[i]) for i in range(len(joint_ids))]

    def jointName2id(self, joint_names: list[str]) -> list[int]:
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[i]) for i in range(len(joint_names))]

    def _clamp_control(self, joint_id, pos) -> float:
        joint_range = self.model.jnt_range[joint_id]
        max_range, min_range = max(joint_range[1], joint_range[0]), min(joint_range[1], joint_range[0])
        clamped_pos = max(min(pos, max_range), min_range)
        return clamped_pos

    def _control_loop(self) -> None:
        while True:
            self._control_joint()
            time.sleep(self.simulation.sim_dt)  # Control loop runs at the visualization timestep
        

    def start_control_loop(self) -> None:
        control_thread = threading.Thread(target=self._control_loop)
        control_thread.start()

    



    