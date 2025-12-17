"""Example: Move to home using Cartesian impedance, then track position with no-friction config."""

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

home_pose = robot.end_effector_pose.copy()
print(f"Starting pose: {robot.end_effector_pose.position}")
print(f"Starting orientation (euler xyz): {robot.end_effector_pose.orientation.as_euler('xyz')}")

# Load the no-friction cartesian impedance configuration
print("\n[3] Loading no-friction configuration...")
robot.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/no_friction_cartesian_impedance.yaml"
)

# Set target to current home position
robot.set_target(pose=home_pose)

print("\n[4] Tracking position with no-friction impedance control...")
print("    (Try gently pushing the robot - it should feel very compliant!)")
print("    Press Ctrl+C to stop.\n")

# Track position in a while loop
try:
    loop_count = 0
    start_time = time.time()
    
    while True:
        current_pose = robot.end_effector_pose
        target_pose = robot.target_pose
        
        # Calculate position error
        position_error = np.linalg.norm(current_pose.position - target_pose.position)
        
        # Print status every 50 iterations (~1 second at 50 Hz)
        if loop_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Current pos: {current_pose.position} | "
                  f"Target pos: {target_pose.position} | "
                  f"Error: {position_error*1000:.2f} mm")
        
        loop_count += 1
        time.sleep(0.02)  # ~50 Hz
        
except KeyboardInterrupt:
    print("\n\n[5] Stopping position tracking...")
    print("Returning to home position...")
    robot.home()
    print("Done!")
    robot.shutdown()
