"""Navigate a single FR3 robot through multiple joint waypoints.

This example demonstrates how to move the robot through a series of predefined
waypoints using joint impedance control. Each waypoint is defined by target
joint angles rather than Cartesian poses.
"""

import time
import numpy as np

from crisp_py.robot import make_robot

# Define joint waypoints (arrays of 7 joint angles in radians)
# Each waypoint specifies the angles for all 7 joints of the FR3 robot
nodes = {
    "Home": np.array([0.5185366,  -0.28910798,  0.33884725, -1.0532444,   0.17898294,  2.271152, 0.78513587]),
    "Transition": np.array([0.02258485,  0.21163805, -0.04094419, -2.71751,     0.03309597,  2.9137363, 0.78436273]),
    "ReadyInsert": np.array([0.02258485,  0.21163805, -0.04094419, -2.71751 ,    0.03309597,  2.9137363, 0.78436273]),
    "FullInsert": np.array([ 0.0230038,   0.3428838,  -0.0359825,  -2.529788,    0.03308011,  2.8802805, 0.78423095]),
    "Pause": {
        "sleep": 10
    }
}

# Configuration parameters
MOVE_TIME = 5.0  # Time to wait for robot to reach each waypoint (seconds)
POSITION_TOLERANCE = 0.05  # Maximum acceptable joint error (radians)
MAX_RETRIES = 3  # Maximum number of retries if waypoint not reached


def wait_for_joint_target(robot, target_joint, tolerance=POSITION_TOLERANCE, timeout=10.0, check_rate=20.0):
    """Wait until the robot reaches the target joint configuration.
    
    Args:
        robot: Robot instance
        target_joint: Target joint configuration (numpy array)
        tolerance: Maximum acceptable joint error in radians
        timeout: Maximum time to wait in seconds
        check_rate: How often to check in Hz
        
    Returns:
        bool: True if target reached within tolerance, False if timeout
    """
    rate = robot.node.create_rate(check_rate)
    elapsed_time = 0.0
    check_period = 1.0 / check_rate
    
    while elapsed_time < timeout:
        current_joint = robot.joint_values
        joint_error = np.linalg.norm(current_joint - target_joint)
        
        if joint_error < tolerance:
            return True
            
        rate.sleep()
        elapsed_time += check_period
    
    return False


# Initialize the robot
robot = make_robot("fr3")
robot.wait_until_ready()

# Move to home position first
print("Moving to home position...")
robot.home()
time.sleep(1.0)

# Configure the joint impedance controller
print("Configuring joint impedance controller...")
robot.controller_switcher_client.switch_controller("joint_impedance_controller")

# Optional: Load joint controller parameters if you have a config file
# robot.joint_controller_parameters_client.load_param_config(
#     file_path="config/control/joint_control.yaml"
# )

print(f"Starting joint configuration: {robot.joint_values}")
print(f"Joint names: {robot.config.joint_names}")


# ======================================================================
# Joint waypoint navigation
# ======================================================================
print("\n" + "="*70)
print("Joint space waypoint navigation")
print("="*70)

waypoint_path = [
    nodes.get("Home"),
    nodes.get("Transition"),
    nodes.get("ReadyInsert"),
    nodes.get("FullInsert"),
    nodes.get("Pause"),
    nodes.get("FullInsert"),
    nodes.get("ReadyInsert"),
    nodes.get("Transition"),
    nodes.get("Home")
]

print(f"\nNavigating through {len(waypoint_path)} waypoints...")

for i, waypoint in enumerate(waypoint_path, 1):
    # Check if waypoint is a dictionary with a sleep command
    if isinstance(waypoint, dict) and "sleep" in waypoint:
        print(f"\n[Waypoint {i}/{len(waypoint_path)}] Pausing for {waypoint['sleep']} seconds...")
        time.sleep(waypoint["sleep"])
        continue

    # Otherwise it's a joint configuration array
    print(f"\n[Waypoint {i}/{len(waypoint_path)}] Moving to joint configuration:")
    print(f"  Target joints: {np.round(waypoint, 3)}")
    
    # Attempt to reach the waypoint with retries
    success = False
    for attempt in range(MAX_RETRIES):
        # Set the target joint configuration
        robot.set_target_joint(waypoint)
        
        # Wait for robot to reach the target
        success = wait_for_joint_target(
            robot, 
            waypoint, 
            tolerance=POSITION_TOLERANCE,
            timeout=MOVE_TIME + 2.0
        )
        
        if success:
            break
        else:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed to reach target within tolerance")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying...")
    
    # Report results
    current_joint = robot.joint_values
    joint_error = np.linalg.norm(current_joint - waypoint)
    max_joint_error = np.max(np.abs(current_joint - waypoint))
    
    if success:
        print(f"  ✓ Reached waypoint {i}")
    else:
        print(f"  ⚠ Failed to reach waypoint {i} within tolerance after {MAX_RETRIES} attempts")
    
    print(f"  Current joints: {np.round(current_joint, 3)}")
    print(f"  Joint error (norm): {joint_error:.4f} rad")
    print(f"  Max joint error: {max_joint_error:.4f} rad")
    
    # Brief pause before next waypoint
    time.sleep(0.5)

print("\n" + "="*70)
print("✓ All waypoints completed!")
print("="*70)

# Return to home position
print("\nReturning to home position...")
robot.home()
time.sleep(1.0)

# Shutdown the robot
print("Shutting down...")
robot.shutdown()
print("Done!")
