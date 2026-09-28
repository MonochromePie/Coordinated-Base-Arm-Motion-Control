from walker import Walker
from walkerSimulation import WalkerSimulation
from walkerControl import WalkerControl
import time


def main():
    xml_path = "simulation/walker_scene.xml"
    walker = Walker(xml_path)
    sim = WalkerSimulation(walker, sim_dt=0.002, vis_dt=0.01, showTime=True)
    control = WalkerControl(sim)

    sim.visualize()
    start_time = walker.data.time
    pos_called = False
    while True:
        elapsed_time = walker.data.time - start_time

        if elapsed_time < 2.0:
            control.set_joint_velocity([walker.joints_info["openarm_left_joint1"]], [-0.5])
        elif elapsed_time < 4.0:

            # Should joint should be closed to -1.0 after 2 seconds of moving at -0.5 rad/s
            if not pos_called:
                print("Targeted joint position:", control._targeted_joint_positions[walker.joints_info["openarm_left_joint1"]])
                time.sleep(1.0)
                control.set_joint_position([walker.joints_info["openarm_left_joint1"], walker.joints_info["x"]], [-2.0,1.0], vel=1.5, acc=3.0)
                pos_called = True

        elif elapsed_time < 8.0:
            control.set_joint_velocity([walker.joints_info["openarm_left_joint1"]], [0.5])
        #should be back to 0.0 after complete

        control.step()
        time.sleep(0.01)



if __name__ == "__main__":
    main()