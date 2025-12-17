"""Navigate a single FR3 robot through multiple waypoints.

This example demonstrates how to move the robot through a series of predefined
waypoints using Cartesian impedance control.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import threading

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose
from scipy.spatial.transform import Rotation

# Define waypoints with both position and orientation
nodes = {
    "Storage":
        Pose(
        position=np.array([0.0, 0.09, 0.35]),
        orientation=Rotation.from_euler('xyz', [-180.0,  52.1,    90.1], degrees=True)
        ),
    "Transition0":
        Pose(
        position=np.array([0.2, 0.10, 0.35]),
        orientation=Rotation.from_euler('xyz', [-180.0,  52.1,    90.1], degrees=True)
        ),
    "Transition":
        Pose(
        position=np.array([0.30, 0.0,  0.48]),
        orientation=Rotation.from_euler('xyz', [-180, -2.51149549e-05, -4.757517], degrees=True)
        ),
    "Transition2":
        Pose(
        position=np.array([0.30, -0.00138537,  0.04610078]),
        orientation=Rotation.from_euler('xyz', [-180,  0.00461875, -4.757517], degrees=True)
        ),
    "ReadyInsert":
        Pose(
        position=np.array([  0.40475205, -0.00513635,  0.04563937]),
        orientation=Rotation.from_euler('xyz', [-1.79932687e+02,  0.00461875 ,-4.757517], degrees=True)
        ),
    "FullInsert":
        Pose(
        position=np.array([0.45475205, -0.00513635,  0.04563937]),
        orientation=Rotation.from_euler('xyz', [-1.79932687e+02, 0.00461875 ,-4.757517], degrees=True)
        ),
    "Pause": {
        "sleep": 5
    }
}


speed = 0.1

# ======================================================================
# Trajectory data collection
# ======================================================================
class TrajectoryData:
    """Class to store trajectory data during robot motion"""
    def __init__(self):
        self.times = []
        self.actual_positions = []
        self.target_positions = []
        # Joint space data (for 7 joints)
        self.joint_positions = []
        self.recording = False
        self.start_time = None
        self.current_target = None
        self.lock = threading.Lock()
    
    def start_recording(self, target_position):
        """Start recording trajectory data"""
        with self.lock:
            self.recording = True
            self.start_time = time.time()
            self.current_target = target_position.copy()
    
    def stop_recording(self):
        """Stop recording trajectory data"""
        with self.lock:
            self.recording = False
    
    def update_target(self, target_position):
        """Update the current target position"""
        with self.lock:
            self.current_target = target_position.copy()
    
    def record_point(self, actual_position, joint_pos=None):
        """Record a single data point"""
        with self.lock:
            if self.recording and self.current_target is not None:
                elapsed = time.time() - self.start_time
                self.times.append(elapsed)
                self.actual_positions.append(actual_position.copy())
                self.target_positions.append(self.current_target.copy())
                # Record joint data if provided
                if joint_pos is not None:
                    self.joint_positions.append(joint_pos.copy())

def data_collection_thread(robot, trajectory_data, sample_rate=50):
    """Background thread to collect position and joint data at specified rate"""
    period = 1.0 / sample_rate
    while trajectory_data.recording:
        actual_pos = robot.end_effector_pose.position
        joint_pos = robot.joint_values
        trajectory_data.record_point(actual_pos, joint_pos)
        time.sleep(period)

def plot_trajectory(trajectory_data, title="End Effector Trajectory"):
    """Plot the trajectory data in 3 subplots (X, Y, Z)"""
    if len(trajectory_data.times) == 0:
        print("No trajectory data to plot!")
        return
    
    times = np.array(trajectory_data.times)
    actual = np.array(trajectory_data.actual_positions)
    target = np.array(trajectory_data.target_positions)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    labels = ['X', 'Y', 'Z']
    colors_target = ['green', 'green', 'green']
    colors_actual = ['blue', 'blue', 'blue']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, target[:, i], '--', color=colors_target[i], 
                linewidth=2, label=f'Target {label}')
        ax.plot(times, actual[:, i], '-', color=colors_actual[i], 
                linewidth=1.5, label=f'Actual {label}')
        ax.set_ylabel(f'{label} Position (m)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Calculate tracking error
        error = np.abs(actual[:, i] - target[:, i])
        max_error = np.max(error)
        mean_error = np.mean(error)
        ax.set_title(f'{label} Trajectory (Max Error: {max_error:.4f} m, Mean Error: {mean_error:.4f} m)')
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_joint_trajectory(trajectory_data, title="Joint Space Trajectory"):
    """Plot joint position trajectory data (7 joints)"""
    if len(trajectory_data.times) == 0 or len(trajectory_data.joint_positions) == 0:
        print("No joint trajectory data to plot!")
        return
    
    times = np.array(trajectory_data.times)
    joint_positions = np.array(trajectory_data.joint_positions)
    
    # Create 7 rows × 1 column subplot (one row per joint)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot each joint position
    for joint_idx in range(7):
        ax = axes[joint_idx]
        ax.plot(times, joint_positions[:, joint_idx], 'b-', linewidth=1.5)
        ax.set_title(f'Joint {joint_idx} Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position (rad)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom subplot
        if joint_idx == 6:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Initialize the robot
robot = make_robot("fr3")
robot.wait_until_ready()

# # Move to home position first
# print("Moving to home position...")
# robot.home()
# time.sleep(1.0)

# Configure the cartesian impedance controller
print("Configuring controller...")
robot.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/clipped_cartesian_impedance.yaml"
)
robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
robot.set_target(pose=robot.end_effector_pose)  # Ensure target is set to current pose
time.sleep(2.0)  # Increased sleep for stability

print(f"Starting pose: {robot.end_effector_pose.position}")
print(f"Starting orientation (euler xyz): {robot.end_effector_pose.orientation.as_euler('xyz')}")


print("\n" + "="*70)
print("Full pose waypoints with orientation control")
print("="*70)

# Initialize trajectory data collection
trajectory_data = TrajectoryData()
collection_thread = None

waypoint_path = [
    # APPROACH HOLE
    nodes.get("Storage"),
    nodes.get("Pause"),
    nodes.get("Transition0"),
    nodes.get("Transition"),
    nodes.get("Transition2"),
    nodes.get("ReadyInsert"),
    nodes.get("Pause"),
    # HOLE INSERTION
    {"switch_config_compliant": "config/control/gravity_compensation_peginhole.yaml"},
    nodes.get("FullInsert"),
    nodes.get("Pause"),
    nodes.get("ReadyInsert"),
    # RETRACTION
    {"switch_config": "config/control/clipped_cartesian_impedance.yaml"},
    nodes.get("Transition2"),
    nodes.get("Transition"),
    nodes.get("Transition0"),
    nodes.get("Storage"),
]

print(f"\nNavigating through {len(waypoint_path)} full pose waypoints...")

# Start trajectory recording before approach
print("Starting trajectory data collection...")
trajectory_data.start_recording(waypoint_path[0].position)
collection_thread = threading.Thread(
    target=data_collection_thread,
    args=(robot, trajectory_data),
    daemon=True
)
collection_thread.start()

for i, waypoint in enumerate(waypoint_path, 1):

    # Check if waypoint is a dictionary with a command
    if isinstance(waypoint, dict):
        if "sleep" in waypoint:
            print(f"Pausing for {waypoint['sleep']} seconds...")
            time.sleep(waypoint["sleep"])
            continue
        elif "switch_config_compliant" in waypoint:
            config_path = waypoint["switch_config_compliant"]
            print(f"Switching to config: {config_path}")
            robot.cartesian_controller_parameters_client.load_param_config(file_path=config_path)

            # Maintain current joint configuration in nullspace
            robot.set_target_joint(robot.joint_values)

            # Apply 1N force in X direction (insertion direction)
            print("Applying 1N force in X direction (insertion direction)...")
            robot.set_target_wrench(force=np.array([2.0, 0.0, 0.0]), torque=np.array([0.0, 0.0, 0.0]))
            
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
    
    # Update target position for trajectory tracking
    trajectory_data.update_target(waypoint.position)
    
    robot.move_to(pose=waypoint, speed=speed)
    time.sleep(0.5)
    
    current_pos = robot.end_effector_pose.position
    current_euler = robot.end_effector_pose.orientation.as_euler('xyz')
    pos_error = np.linalg.norm(current_pos - waypoint.position)
    ori_error = np.linalg.norm(current_euler - euler)
    print(f"  Reached waypoint {i}. Position error: {pos_error:.4f} m, Orientation error: {ori_error:.4f} rad")

    # Clear force after insertion completes
    if waypoint is nodes["FullInsert"]:
        print("Clearing insertion force...")
        robot.set_target_wrench(force=np.array([0.0, 0.0, 0.0]), torque=np.array([0.0, 0.0, 0.0]))



print("\n✓ All waypoints completed!")




print("Stopping trajectory data collection...")
trajectory_data.stop_recording()
if collection_thread:
    collection_thread.join(timeout=1.0)
print(f"Collected {len(trajectory_data.times)} data points")

# Return to home position
# print("\nReturning to home position...")
# robot.home()
# time.sleep(1.0)

# Shutdown the robot
print("Shutting down...")
robot.shutdown()
print("Done!")

# Display trajectory plots
print("\n" + "="*70)
print("Displaying trajectory plots...")
print("="*70)
plot_trajectory(trajectory_data, title="Peg-in-Hole Insertion Trajectory")
print("\nDisplaying joint space trajectory...")
plot_joint_trajectory(trajectory_data, title="Joint Space Trajectory")
