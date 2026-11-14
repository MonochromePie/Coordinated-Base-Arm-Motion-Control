"""Simulation-only code: MuJoCo backend, target GUI and the loop that ties them to the IK."""
from .mujoco_sim import IK_TO_SIM_OFFSET, MujocoSim
from .target_gui import TargetGUI
from .control_loop import run_control_loop
from .drag_targets import DragTargets

__all__ = ["DragTargets", "IK_TO_SIM_OFFSET", "MujocoSim", "TargetGUI", "run_control_loop"]
