"""Try to follow a "figure eight" target on the yz plane."""

# %%
import matplotlib.pyplot as plt
import numpy as np

from crisp_py.robot import make_robot

from crisp_py.robot import Robot, FrankaConfig
faster_publishing_config = FrankaConfig()
faster_publishing_config.publish_frequency = 20.0
left_arm = Robot(robot_config=faster_publishing_config, namespace="left")
right_arm = Robot(robot_config=faster_publishing_config, namespace="right")

left_arm.wait_until_ready()
right_arm.wait_until_ready()

# %%
print("Left arm:", left_arm.end_effector_pose)
print("Left arm joints:", left_arm.joint_values)
print("Right arm:", right_arm.end_effector_pose)
print("Right arm joints:", right_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()
right_arm.home()
left_homing_pose = left_arm.end_effector_pose.copy()
right_homing_pose = right_arm.end_effector_pose.copy()


# %%
# Paremeters for the circle
radius = 0.2  # [m]
left_center = np.array([0.4, 0.0, 0.4])
right_center = np.array([0.4, -0.4, 0.4])  # Offset in -y direction for spatial separation
ctrl_freq = 50.0
sin_freq_y = 0.25  # rot / s
sin_freq_z = 0.125  # rot / s
max_time = 8.0

# %%
left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/default_cartesian_impedance.yaml"
)
right_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
right_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/default_cartesian_impedance.yaml"
)

# %%
# The move_to function will publish a pose to /target_pose while interpolation linearly
print("Moving both arms to their starting centers...")
left_arm.move_to(position=left_center, speed=0.15)
right_arm.move_to(position=right_center, speed=0.15)

# %%
# The set_target will directly publish the pose to /target_pose
left_ee_poses = []
left_target_poses = []
right_ee_poses = []
right_target_poses = []
ts = []

print("Starting to draw figure eights with both arms...")
t = 0.0
left_target_pose = left_arm.end_effector_pose.copy()
right_target_pose = right_arm.end_effector_pose.copy()
rate = left_arm.node.create_rate(ctrl_freq)

while t < max_time:
    # Left arm trajectory
    x_left = left_center[0]
    y_left = radius * np.sin(2 * np.pi * sin_freq_y * t) + left_center[1]
    z_left = radius * np.sin(2 * np.pi * sin_freq_z * t) + left_center[2]
    left_target_pose.position = np.array([x_left, y_left, z_left])
    
    # Right arm trajectory
    x_right = right_center[0]
    y_right = radius * np.sin(2 * np.pi * sin_freq_y * t) + right_center[1]
    z_right = radius * np.sin(2 * np.pi * sin_freq_z * t) + right_center[2]
    right_target_pose.position = np.array([x_right, y_right, z_right])

    left_arm.set_target(pose=left_target_pose)
    right_arm.set_target(pose=right_target_pose)

    rate.sleep()

    left_ee_poses.append(left_arm.end_effector_pose.copy())
    left_target_poses.append(left_arm._target_pose.copy())
    right_ee_poses.append(right_arm.end_effector_pose.copy())
    right_target_poses.append(right_arm._target_pose.copy())
    ts.append(t)

    t += 1.0 / ctrl_freq

while t < max_time + 1.0:
    # Just wait a bit for the end effectors to settle

    rate.sleep()

    left_ee_poses.append(left_arm.end_effector_pose.copy())
    left_target_poses.append(left_arm._target_pose.copy())
    right_ee_poses.append(right_arm.end_effector_pose.copy())
    right_target_poses.append(right_arm._target_pose.copy())
    ts.append(t)

    t += 1.0 / ctrl_freq


print("Done drawing figure eights!")


# %%
# Extract position data for left arm
left_y_t = [target_pose_sample.position[1] for target_pose_sample in left_target_poses]
left_z_t = [target_pose_sample.position[2] for target_pose_sample in left_target_poses]
left_y_ee = [ee_pose.position[1] for ee_pose in left_ee_poses]
left_z_ee = [ee_pose.position[2] for ee_pose in left_ee_poses]

# Extract position data for right arm
right_y_t = [target_pose_sample.position[1] for target_pose_sample in right_target_poses]
right_z_t = [target_pose_sample.position[2] for target_pose_sample in right_target_poses]
right_y_ee = [ee_pose.position[1] for ee_pose in right_ee_poses]
right_z_ee = [ee_pose.position[2] for ee_pose in right_ee_poses]

# %%
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Left arm YZ trajectory
ax[0, 0].plot(left_y_ee, left_z_ee, label="Left arm current", color='blue', linewidth=2)
ax[0, 0].plot(left_y_t, left_z_t, label="Left arm target", linestyle="--", color='blue', alpha=0.6)
ax[0, 0].set_xlabel("$y$ [m]")
ax[0, 0].set_ylabel("$z$ [m]")
ax[0, 0].set_title("Left Arm Figure Eight")
ax[0, 0].legend()
ax[0, 0].grid()

# Top-right: Right arm YZ trajectory
ax[0, 1].plot(right_y_ee, right_z_ee, label="Right arm current", color='orange', linewidth=2)
ax[0, 1].plot(right_y_t, right_z_t, label="Right arm target", linestyle="--", color='orange', alpha=0.6)
ax[0, 1].set_xlabel("$y$ [m]")
ax[0, 1].set_ylabel("$z$ [m]")
ax[0, 1].set_title("Right Arm Figure Eight")
ax[0, 1].legend()
ax[0, 1].grid()

# Bottom-left: Time series for z-position
ax[1, 0].plot(ts, left_z_ee, label="Left arm current", color='blue', linewidth=2)
ax[1, 0].plot(ts, left_z_t, label="Left arm target", linestyle="--", color='blue', alpha=0.6)
ax[1, 0].plot(ts, right_z_ee, label="Right arm current", color='orange', linewidth=2)
ax[1, 0].plot(ts, right_z_t, label="Right arm target", linestyle="--", color='orange', alpha=0.6)
ax[1, 0].set_xlabel("$t$ [s]")
ax[1, 0].set_ylabel("$z$ [m]")
ax[1, 0].set_title("Z-Position vs Time")
ax[1, 0].legend()
ax[1, 0].grid()

# Bottom-right: Both arms YZ trajectories overlaid
ax[1, 1].plot(left_y_ee, left_z_ee, label="Left arm", color='blue', linewidth=2)
ax[1, 1].plot(right_y_ee, right_z_ee, label="Right arm", color='orange', linewidth=2)
ax[1, 1].set_xlabel("$y$ [m]")
ax[1, 1].set_ylabel("$z$ [m]")
ax[1, 1].set_title("Both Arms (Spatial View)")
ax[1, 1].legend()
ax[1, 1].grid()

fig.tight_layout()
plt.show()

# %%
print("Going back home.")
left_arm.home()
right_arm.home()

# %%
left_arm.shutdown()
right_arm.shutdown()
