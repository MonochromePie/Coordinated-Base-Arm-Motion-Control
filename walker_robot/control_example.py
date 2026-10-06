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
            
            
        # joint should be closed to -1.0 after 2 seconds of moving at -0.5 rad/s 
        elif elapsed_time < 4.0:

            #position control only called once
            if not pos_called:
                time.sleep(1.0)
                control.set_joint_position([walker.joints_info["openarm_right_joint1"], walker.joints_info["x"], walker.joints_info["y"], walker.joints_info["lift"], walker.joints_info["yaw"]], [2.0,1.0,1.0,1.0,1.5], vel=5.0, acc=10.0, sync=True)
            
                pos_called = True
                print(walker.joints_info)

        elif elapsed_time < 8.0:
            control.set_joint_velocity([walker.joints_info["openarm_left_joint1"]], [0.25])

        # joint should be closed to 0.0 after 2 seconds of moving at -0.25 rad/s because no shit why wouldnt it not

        control.step()
        time.sleep(0.01)



if __name__ == "__main__":
    main()