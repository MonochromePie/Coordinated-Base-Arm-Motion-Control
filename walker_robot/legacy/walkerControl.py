from walkerSimulation import WalkerSimulation
from walker import Walker, WalkerInfo
from dataclasses import dataclass
import mujoco


@dataclass
class TrapezoidalProfile:
    start_pos: float
    target_pos: float
    vel: float
    acc: float
    t0: float

    def __post_init__(self):
        if self.vel <= 0 or self.acc <= 0:
            raise ValueError("vel and acc must be positive.")

        delta = self.target_pos - self.start_pos
        self.distance = abs(delta)
        self.sign = 1.0 if delta >= 0 else -1.0

        if self.distance < self.vel ** 2 / self.acc:      # triangular
            self.t_acc = (self.distance / self.acc) ** 0.5
            self.v_peak = self.acc * self.t_acc
            self.t_const = 0.0
        else:                                             # trapezoidal
            self.t_acc = self.vel / self.acc
            self.v_peak = self.vel
            self.t_const = self.distance / self.vel - self.vel / self.acc

        self.total_time = 2 * self.t_acc + self.t_const

    def distance_at(self, t: float) -> float:
        """Distance travelled from start (>= 0) after t seconds."""
        a, ta, tc, T, d = self.acc, self.t_acc, self.t_const, self.total_time, self.distance
        if t <= 0:
            return 0.0
        if t >= T:
            return d
        if t < ta:
            return 0.5 * a * t ** 2
        if t < ta + tc:
            return 0.5 * a * ta ** 2 + self.v_peak * (t - ta)
        return d - 0.5 * a * (T - t) ** 2

    def position_at(self, t: float) -> float:
        return self.start_pos + self.sign * self.distance_at(t)

    def is_done(self, t: float) -> bool:
        return t >= self.total_time


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

        # active trapezoidal profiles, keyed by joint id
        self._profiles: dict[int, TrapezoidalProfile] = {}

        # joint id -> actuator id (built once instead of searched every step)
        self._joint_actuator: dict[int, int | None] = {
            joint_id: self._find_actuator(joint_id) for joint_id in self._targeted_joint_positions
        }

    def step(self) -> None:
        now = self.data.time
        self._control_dt = now - self._last_control_time
        self._last_control_time = now

        # advance active profiles using simulation time
        for joint_id, profile in list(self._profiles.items()):
            elapsed = now - profile.t0
            self._targeted_joint_positions[joint_id] = profile.position_at(elapsed)
            if profile.is_done(elapsed):
                del self._profiles[joint_id]

        # write commands
        for joint_id, target in self._targeted_joint_positions.items():
            actuator_id = self._joint_actuator.get(joint_id)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = self._clamp_control(joint_id, target)

    # ------------------------------------------------------------ public API
    def set_joint(self, joint_ids: list[int], pos: list[float]) -> None:
        """Jump-command joints to a position (cancels any active profile)."""
        if len(joint_ids) != len(pos):
            raise ValueError("Length of joint_ids and pos must be the same.")

        for joint_id, target_pos in zip(joint_ids, pos):
            self._interrupt_profile(joint_id)
            self._targeted_joint_positions[joint_id] = target_pos

    def set_joint_velocity(self, joint_ids: list[int], joint_vel: list[float]) -> None:
        if len(joint_ids) != len(joint_vel):
            raise ValueError("Length of joint_ids and joint_vel must be the same.")

        for joint_id, vel in zip(joint_ids, joint_vel):
            self._interrupt_profile(joint_id)
            self._targeted_joint_positions[joint_id] += vel * self._control_dt

    def set_joint_position(self, joint_ids: list[int], target_pos: list[float],
                           vel: float = 0.5, acc: float = 0.2, sync: bool = False) -> None:
        """Move joints to target positions with a trapezoidal velocity profile.

        vel / acc are per-joint limits. If sync=True, all joints in this call are
        time-scaled so they arrive at the same moment (the slowest joint sets the pace).
        """
        if len(joint_ids) != len(target_pos):
            raise ValueError("Length of joint_ids and target_pos must be the same.")

        now = self.data.time
        profiles: dict[int, TrapezoidalProfile] = {}
        for joint_id, pos in zip(joint_ids, target_pos):
            profiles[joint_id] = TrapezoidalProfile(
                start_pos=self._targeted_joint_positions[joint_id],   # last commanded, not measured qpos
                target_pos=self._clamp_control(joint_id, pos),
                vel=vel,
                acc=acc,
                t0=now,
            )

        if sync and profiles:
            T = max(p.total_time for p in profiles.values())
            if T > 0:
                for joint_id, p in list(profiles.items()):
                    if p.total_time <= 0:
                        continue
                    # stretching time by k: vel -> vel/k, acc -> acc/k^2, duration -> duration*k
                    k = T / p.total_time
                    profiles[joint_id] = TrapezoidalProfile(
                        p.start_pos, p.target_pos, p.v_peak / k, p.acc / k ** 2, now
                    )

        self._profiles.update(profiles)

    # --------------------------------------------------------------- helpers
    def is_moving(self, joint_ids: list[int] | None = None) -> bool:
        """True if any (or any of the given) joints still have an active profile."""
        if joint_ids is None:
            return bool(self._profiles)
        return any(j in self._profiles for j in joint_ids)

    def joint2actuatorID(self, joint_id: int) -> int | None:
        return self._joint_actuator.get(joint_id)

    def _find_actuator(self, joint_id: int) -> int | None:
        for actuator_id in range(self.model.nu):
            if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                return actuator_id
        return None

    def jointID2name(self, joint_ids: list[int]) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]

    def jointName2id(self, joint_names: list[str]) -> list[int]:
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]

    def _clamp_control(self, joint_id, pos) -> float:
        joint_range = self.model.jnt_range[joint_id]
        max_range, min_range = max(joint_range[1], joint_range[0]), min(joint_range[1], joint_range[0])
        return max(min(pos, max_range), min_range)

    def _interrupt_profile(self, joint_id: int) -> None:
        self._profiles.pop(joint_id, None)
