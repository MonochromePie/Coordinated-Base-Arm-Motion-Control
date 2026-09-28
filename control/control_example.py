from walker import Walker
from walkerSimulation import WalkerSimulation
from walkerControl import WalkerControl
import time
import numpy as np

def main():
    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    sim = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01, showTime=True)
    control = WalkerControl(sim)

    sim.visualize()

    start_time = walker.data.time
    while True:
        elapsed_time = walker.data.time - start_time

        if elapsed_time < 2.0:
            control.set_joint_velocity([walker.joints_info["openarm_left_joint1"]], [-0.5])
        else:
            print(control._targeted_joint_positions[walker.joints_info["openarm_left_joint1"]])
            sim.close()
            break

        control.step()
        time.sleep(0.01)



if __name__ == "__main__":
    main()