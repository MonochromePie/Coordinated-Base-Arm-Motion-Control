from walker import Walker
from mujoco import viewer
import mujoco
from dataclasses import dataclass
import threading
import time
import numpy as np

# the IK model's base sits 0.1 m lower than MuJoCo's (h = 0.697 - 0.1): markers take IK-model positions
IK_TO_SIM_OFFSET = np.array([0.0, 0.0, 0.1])

@dataclass
class WalkerSimulationInfo:
    walker: Walker
    sim_dt: float
    vis_dt: float

class WalkerSimulation:
    def __init__(self, walker: Walker, sim_dt = 0.002, vis_dt: float = 0.01, showTime: bool = False):
        self.walker = walker
        self.sim_dt = sim_dt
        self.walker.model.opt.timestep = self.sim_dt
        self.vis_dt = vis_dt
        self.walker.vis_dt = self.vis_dt

        self._last_sync_time = 0.0
        self._start_time = 0.0
        self._showTime = showTime
        self._target_markers: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self._target_marker_lock = threading.Lock()
        

        self.sim: viewer.Handle | None = None
        self._sync_thread: threading.Thread | None = None

        self._is_running = False

    def visualize(self) -> None:
        self._start_time = time.time()
        self._is_running = True
        self.sim = viewer.launch_passive(self.walker.model, self.walker.data)
        self._sync_thread = threading.Thread(target=self._sim_sync, daemon=True)
        self._sync_thread.start()

    def _sim_sync(self) -> None:
        sim = self.sim                      # local reference, safe from close()
        while self._is_running and sim is not None and sim.is_running():
            current_time = time.time()
            if current_time - self._last_sync_time >= self.vis_dt:
                self._step_simulation(sim)
                self._update_target_marker(sim)
                sim.sync()
                if self._showTime:
                    print(f"Simulation Time: {self.walker.data.time:.2f} seconds")
                self._last_sync_time = current_time
            else:
                time.sleep(0.001)
        self._is_running = False

    def set_target_marker(self, position: np.ndarray, name: str = "target",
                          rgba=(1.0, 0.0, 0.0, 1.0), radius: float = 0.03) -> None:
        """Show a sphere at a world-space position of the IK model (drawn IK_TO_SIM_OFFSET higher).
        Markers are keyed by name, so calling again with the same name moves that marker."""
        with self._target_marker_lock:
            self._target_markers[name] = (np.asarray(position, dtype=float) + IK_TO_SIM_OFFSET,
                                          np.asarray(rgba, dtype=np.float32),
                                          float(radius))

    def clear_target_markers(self) -> None:
        with self._target_marker_lock:
            self._target_markers.clear()

    def _update_target_marker(self, sim) -> None:
        with self._target_marker_lock:
            markers = list(self._target_markers.values())

        with sim.lock():
            scn = sim.user_scn
            n = min(len(markers), scn.maxgeom)
            for i, (position, rgba, radius) in enumerate(markers[:n]):
                mujoco.mjv_initGeom(
                    scn.geoms[i],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([radius, radius, radius]),
                    position,
                    np.eye(3).ravel(),
                    rgba,
                )
            scn.ngeom = n


    def _step_simulation(self, sim) -> None:
        steps = int(self.vis_dt / self.sim_dt)
        for _ in range(steps):
            mujoco.mj_step(self.walker.model, self.walker.data)

    def close(self) -> None:
        self._is_running = False          
        if self._sync_thread is not None and self._sync_thread is not threading.current_thread():
            self._sync_thread.join()
            self._sync_thread = None
        if self.sim is not None:
            self.sim.close()
            self.sim = None
    
    def get_simulation_info(self) -> WalkerSimulationInfo:
        return WalkerSimulationInfo(
            walker=self.walker,
            sim_dt=self.sim_dt,
            vis_dt=self.vis_dt
        )

