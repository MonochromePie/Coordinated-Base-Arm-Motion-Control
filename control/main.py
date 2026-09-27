from walker import Walker
from walkerSimulation import WalkerSimulation
from walkerControl import WalkerControl
import time
import numpy as np

def main():
    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    simulation = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01, showTime=True)
    control = WalkerControl(simulation)

    simulation.visualize()
    control.start_control_loop()

    time.sleep(2.0)  # Wait for 1 second before starting the control loop
    while True:
        if walker.data.time <= 5.0:

            # print(control._targeted_joint_positions[walker.joints_info["openarm_right_joint1"]])
            control.set_joint_velocity(
                        joint_ids=[walker.joints_info["openarm_right_joint1"]],
                        joint_vel=[0.5]
                    )
        else:
            # print(control._targeted_joint_positions[walker.joints_info["openarm_right_joint1"]])
            break



if __name__ == "__main__":
    main()