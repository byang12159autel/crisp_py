from crisp_py.robot import make_robot
import time
import numpy as np

robot = make_robot("fr3")
robot.wait_until_ready()
robot.home()
time.sleep(1.0)

nodes = {
    "stowaway_joint_config": 
        np.array([np.pi/2, -1.2862661,  -0.01637701 ,-2.3484082,   0.04115199,  2.624527,  0.83065337]),
}
robot.controller_switcher_client.switch_controller("joint_trajectory_controller")

# Send the joint configuration command
robot.joint_trajectory_controller_client.send_joint_config(
    joint_names=robot.config.joint_names,
    joint_config=nodes.get("stowaway_joint_config"),
    time_to_goal=5.0,  # 5 seconds to reach home (same as robot.config.time_to_home)
    blocking=True         # Wait until motion completes
)
# Wait for robot to be ready again
robot.wait_until_ready()
time.sleep(3.0)

robot.shutdown()