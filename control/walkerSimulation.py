from walker import Walker
from mujoco import viewer
import mujoco
from dataclasses import dataclass
import threading
import time

@dataclass
class WalkerSimulationInfo:
    walker: Walker
    sim_dt: float
    vis_dt: float

class WalkerSimulation:
    def __init__(self, walker: Walker, sim_dt = 0.002, vis_dt: float = 0.01):
        self.walker = walker
        self.sim_dt = sim_dt
        self.walker.model.opt.timestep = self.sim_dt
        self.vis_dt = vis_dt
        self.walker.vis_dt = self.vis_dt

        self.sim: viewer.Handle | None = None
        self._last_sync_time = 0.0
        self._start_time = 0.0

    def visualize(self) -> None:
        self._start_time = time.time()
        self.sim = viewer.launch_passive(self.walker.model, self.walker.data)
        sync_thread = threading.Thread(target=self._sim_sync)
        sync_thread.start()

    def _sim_sync(self) -> None:
        while True:
            if self.sim is not None and self.sim.is_running():
                current_time = time.time()
                if current_time - self._last_sync_time >= self.vis_dt:
                    self._step_simularions()
                    self.sim.sync()
                    print(f"Simulation Time: {current_time - self._start_time:.2f} seconds")
                    self._last_sync_time = current_time

    def get_simulation_info(self) -> WalkerSimulationInfo:
        return WalkerSimulationInfo(
            walker=self.walker,
            sim_dt=self.sim_dt,
            vis_dt=self.vis_dt
        )

    def _step_simularions(self) -> None:
        if self.sim is not None and self.sim.is_running():
            steps = int(self.vis_dt / self.sim_dt)
            for _ in range(steps):
                mujoco.mj_step(self.walker.model, self.walker.data)

    def __del__(self):
        if self.sim is not None and not self.sim.is_running():
            self.sim.close()
    
if __name__ == "__main__":

    from walker import Walker
    
    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    simulation = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01)
    simulation.visualize()

    
