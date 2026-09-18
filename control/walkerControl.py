from walkerSimulation import WalkerSimulation
import mujoco
import time

class WalkerControl:
    def __init__(self, walkerSimulation: WalkerSimulation):
        self.simulation = walkerSimulation
        self.walker = self.simulation.walker
        self.walkerInfo = self.walker.get_walker_info()
        self.data = self.walker.data
        self.model = self.walker.model

        self._targeted_joint_positions = self.walker.get_joint_positions()
        self._last_control_time = time.time()

    def _set_joint(self, joint_ids: list[int] , pos: list[float]) -> None:
        if len(joint_ids) != len(pos):
            raise ValueError("Length of joint_ids and pos must be the same.")

        for i, joint_id in enumerate(joint_ids):
            self.data.ctrl[joint_id] = pos[i]

    def set_joint_velocity(self, joint_ids: list[int], joint_vel: list[float]) -> None:
        
        vis_dt = self.simulation.vis_dt
        self._targeted_joint_positions = self.walker.get_joint_positions()
        
        for joint_id, vel in zip(joint_ids, joint_vel):
            current_time = time.time()
            added_pos = vel * vis_dt

            if current_time - self._last_control_time >= vis_dt:
                self._last_control_time = current_time
            else:
                added_pos = 0.0
            self._targeted_joint_positions[joint_id] += added_pos

        # self._clamp_control()
        self._set_joint(joint_ids, [self._targeted_joint_positions[joint_id] for joint_id in joint_ids])
        

    def _jointID2name(self, joint_ids: list[int]) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_ids[i]) for i in range(len(joint_ids))]

    def _jointName2id(self, joint_names: list[str]) -> list[int]:
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[i]) for i in range(len(joint_names))]

    def _clamp_control(self) -> None:
        for joint_id, pos in enumerate(self._targeted_joint_positions.values()):
            joint_range = self.model.jnt_range[joint_id]
            max_range, min_range = max(joint_range[1], joint_range[0]), min(joint_range[1], joint_range[0])
            clamped_pos = max(min(pos, max_range), min_range)
            self._targeted_joint_positions[joint_id] = clamped_pos

            

if __name__ == "__main__":
    from walkerSimulation import WalkerSimulation
    from walker import Walker
    import time

    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    simulation = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01)
    control = WalkerControl(simulation)

    simulation.visualize()
    control._set_joint([0,1,2],[0.5,0.5,0.5]) # Example of setting the first joint to position 0.5 

    time.sleep(2)  # Wait for a second to observe the change in position
    print(walker.get_joint_positions())  # Print the current joint positions
    while True:
        control.set_joint_velocity([0,1,2],[0.1,0.1,0.1])
        time.sleep(0.01)  # while loop delay

    