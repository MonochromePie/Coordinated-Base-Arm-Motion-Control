"""
Bimanual IK demo: MuJoCo viewer + Tkinter target GUI.

The IK (walker_robot/bimanual_ik.py) moves both OpenArm arms, and optionally the Walkie base
(x/y, yaw, lift), towards the targets set with the GUI sliders.

Run:  uv run run_ik_gui.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "walker_robot"))   # walker_robot/ modules import each other flat (from IK import ...)

import numpy as np

from IK import ForwardKinematics
from bimanual_ik import JOINT_LIMITS, JOINT_NAMES, BimanualIK, Target
from sim import MujocoSim, TargetGUI, run_control_loop

SCENE = ROOT / "simulation" / "walker_scene.xml"


def main():
    print("Building IK model (symbolic, takes a few seconds)...")
    ik = BimanualIK()

    sim = MujocoSim(str(SCENE), JOINT_NAMES)
    # all joints at 0, clipped into the joint limits (lift's minimum is 0.1); also the GUI's reset pose
    home_state = {name: float(np.clip(0.0, *JOINT_LIMITS[name])) for name in JOINT_NAMES}
    sim.set_joint_state(home_state)
    start_poses = ik.forward(home_state)
    gui = TargetGUI({side: Target(pos, ForwardKinematics.rotmat_to_euler_ZYX(R))
                     for side, (pos, R) in start_poses.items()})

    run_control_loop(ik, sim, gui, home_state)   # control thread in the background, GUI in the main thread


if __name__ == "__main__":
    main()
