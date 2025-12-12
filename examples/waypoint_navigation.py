"""Navigate a single FR3 robot through multiple waypoints.

This example demonstrates how to move the robot through a series of predefined
waypoints using Cartesian impedance control.
"""

import time
import numpy as np

from crisp_py.robot import make_robot

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

print(f"Starting position: {robot.end_effector_pose.position}")
# Starting position: [3.06892619e-01 1.45795419e-10 4.86878220e-01]

# Define waypoints (x, y, z) coordinates
# These are example positions relative to the robot base
waypoints = [
    np.array([0.5, 0.2, 0.4]),   # Waypoint 1: forward right, mid-height
    np.array([0.5, -0.2, 0.4]),  # Waypoint 2: forward left, mid-height
    np.array([0.4, 0.0, 0.5]),   # Waypoint 3: center, higher
    np.array([0.6, 0.0, 0.3]),   # Waypoint 4: further forward, lower
]

# Movement speed (m/s)
speed = 0.1

# Navigate through each waypoint
print(f"\nStarting waypoint navigation through {len(waypoints)} points...")
for i, waypoint in enumerate(waypoints, 1):
    print(f"\nMoving to waypoint {i}/{len(waypoints)}: {waypoint}")
    robot.move_to(position=waypoint, speed=speed)
    
    # Brief pause at each waypoint
    time.sleep(0.5)
    
    current_pos = robot.end_effector_pose.position
    error = np.linalg.norm(current_pos - waypoint)
    print(f"  Reached waypoint {i}. Position error: {error:.4f} m")

print("\n✓ All waypoints completed!")

# Return to home position
print("\nReturning to home position...")
robot.home()
time.sleep(1.0)

# Shutdown the robot
print("Shutting down...")
robot.shutdown()
print("Done!")
