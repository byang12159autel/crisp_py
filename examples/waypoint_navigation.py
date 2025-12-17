"""Navigate a single FR3 robot through multiple waypoints.

This example demonstrates how to move the robot through a series of predefined
waypoints using Cartesian impedance control.
"""

import time
import numpy as np

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose
from scipy.spatial.transform import Rotation

# Define waypoints with both position and orientation
nodes = {
    "Home":    
        Pose(
        position=np.array([0.21410979, 0.37285984, 1.05776812]),
        orientation=Rotation.from_euler('xyz', [-159.52877581,  -84.8988165,    35.25627904])
        ),
    "Transition":
        Pose(
        position=np.array([0.39329999, -0.01051764,  0.05499903]),
        orientation=Rotation.from_euler('xyz', [179.90851749,   0.86408047,  -2.79495816])
        ),
    "ReadyInsert":    
        Pose(
        position=np.array([ 0.55 ,-1.20004051e-05,  4.86894885e-01]),
        orientation=Rotation.from_euler('xyz', [-np.pi, -2.51149549e-05, -1.32601437e-04])
        ),
    "FullInsert":
        Pose(
        position=np.array([0.70, -1.20004051e-05,  4.86894885e-01]),
        orientation=Rotation.from_euler('xyz', [-np.pi, -2.51149549e-05, -1.32601437e-04])
        ),
    "Pause": {
        "sleep": 5
    }
}


speed = 0.1



# Initialize the robot
robot = make_robot("fr3")
robot.wait_until_ready()

# Move to home position first
print("Moving to home position...")
robot.home()
time.sleep(1.0)

# Configure the cartesian impedance controller
print("Configuring controller...")
robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
robot.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/default_cartesian_impedance.yaml"
)

print(f"Starting pose: {robot.end_effector_pose.position}")
print(f"Starting orientation (euler xyz): {robot.end_effector_pose.orientation.as_euler('xyz')}")


# ======================================================================
# Full pose waypoints (position + orientation)
# ======================================================================
print("\n" + "="*70)
print("Full pose waypoints with orientation control")
print("="*70)

waypoint_path = [
    nodes.get("ReadyInsert"),
    {"switch_config": "config/control/gravity_compensation_peginhole.yaml"},
    nodes.get("FullInsert"),
    nodes.get("Pause"),
]


print(f"\nNavigating through {len(waypoint_path)} full pose waypoints...")
for i, waypoint in enumerate(waypoint_path, 1):

    # Check if waypoint is a dictionary with a command
    if isinstance(waypoint, dict):
        if "sleep" in waypoint:
            print(f"Pausing for {waypoint['sleep']} seconds...")
            time.sleep(waypoint["sleep"])
            continue
        elif "switch_config" in waypoint:
            config_path = waypoint["switch_config"]
            print(f"Switching to config: {config_path}")
            robot.cartesian_controller_parameters_client.load_param_config(file_path=config_path)
            continue

    # Otherwise it's a Pose object
    euler = waypoint.orientation.as_euler('xyz')
    print(f"\n  Moving to waypoint {i}/{len(waypoint_path)}:")
    print(f"    Position: {waypoint.position}")
    print(f"    Orientation (euler xyz): {euler}")
    
    robot.move_to(pose=waypoint, speed=speed)
    time.sleep(0.5)
    
    current_pos = robot.end_effector_pose.position
    current_euler = robot.end_effector_pose.orientation.as_euler('xyz')
    pos_error = np.linalg.norm(current_pos - waypoint.position)
    ori_error = np.linalg.norm(current_euler - euler)
    print(f"  Reached waypoint {i}. Position error: {pos_error:.4f} m, Orientation error: {ori_error:.4f} rad")

print("\n✓ All waypoints completed!")

# Return to home position
print("\nReturning to home position...")
robot.home()
time.sleep(1.0)

# Shutdown the robot
print("Shutting down...")
robot.shutdown()
print("Done!")
