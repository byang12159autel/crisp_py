"""Try to follow a "figure eight" target on the yz plane."""

# %%
import matplotlib.pyplot as plt
import numpy as np

from crisp_py.robot import make_robot

from crisp_py.robot import Robot, FrankaConfig
faster_publishing_config = FrankaConfig()
faster_publishing_config.publish_frequency = 20.0
faster_publishing_config.use_prefix = True  # Required for dual arm with namespaces
left_arm = Robot(robot_config=faster_publishing_config, namespace="left")
right_arm = Robot(robot_config=faster_publishing_config, namespace="right")

print(left_arm._current_joint)
left_arm.wait_until_ready()
right_arm.wait_until_ready()

# %%
print("Left arm pose:", left_arm.end_effector_pose)
print("Left arm joints:", left_arm.joint_values)
print("Right arm pose:", right_arm.end_effector_pose)
print("Right arm joints:", right_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()
right_arm.home()
homing_pose_left = np.array(left_arm.end_effector_pose.position)
homing_pose_right = np.array(right_arm.end_effector_pose.position)

# %%
print("Adding 90 degree yaw to base joint of both arms...")
# Switch to joint control temporarily
left_arm.controller_switcher_client.switch_controller("joint_impedance_controller")
right_arm.controller_switcher_client.switch_controller("joint_impedance_controller")

# Get current joint positions and add 90 degrees (π/2 radians) to base joint
left_joints = left_arm.joint_values.copy()
left_joints[0] -= np.pi / 2  # Add 90 degrees to base joint yaw
left_joints[1] = 1.7
left_joints[3] = 0.0

right_joints = right_arm.joint_values.copy()
right_joints[0] += np.pi / 2  # Add 90 degrees to base joint yaw
right_joints[1] = 1.7
right_joints[3] = 0.0

# Move to new joint positions
left_arm.set_target_joint(left_joints)
right_arm.set_target_joint(right_joints)

# Wait for motion to complete
import time
time.sleep(10.0)

left_joints = left_arm.joint_values.copy()
left_joints[0] -= np.pi / 2  # Add 90 degrees to base joint yaw
right_joints = right_arm.joint_values.copy()
right_joints[0] += np.pi / 2  # Add 90 degrees to base joint yaw
left_arm.set_target_joint(left_joints)
right_arm.set_target_joint(right_joints)
time.sleep(10.0)

# %%
# Parameters for the figure eight
radius = 0.2  # [m]

# Calculate midpoint between the two arms to bring them closer
midpoint_y = (homing_pose_left[1] + homing_pose_right[1]) / 2.0
separation = 0.3  # [m] - distance between centers (adjust for more/less collision risk)

# Move both centers toward the midpoint
center_left = homing_pose_left.copy()
center_left[1] = midpoint_y + separation / 2.0  # Shift left arm toward center

center_right = homing_pose_right.copy()
center_right[1] = midpoint_y - separation / 2.0  # Shift right arm toward center

ctrl_freq = 50.0
sin_freq_y = 0.25  # rot / s
sin_freq_z = 0.125  # rot / s
max_time = 8.0

# %%
left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
right_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/default_cartesian_impedance.yaml"
)
right_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/default_cartesian_impedance.yaml"
)

# %%
# The move_to function will publish a pose to /target_pose while interpolation linearly
print("Moving to starting positions...")
left_arm.move_to(position=center_left, speed=0.15)
right_arm.move_to(position=center_right, speed=0.15)

# %%
# The set_target will directly publish the pose to /target_pose
ee_poses_left = []
target_poses_left = []
ee_poses_right = []
target_poses_right = []
ts = []

print("Starting to draw figure eight patterns...")
t = 0.0
target_pose_left = left_arm.end_effector_pose.copy()
target_pose_right = right_arm.end_effector_pose.copy()
rate = left_arm.node.create_rate(ctrl_freq)

while t < max_time:
    # Left arm figure eight
    x_left = center_left[0]
    y_left = radius * np.sin(2 * np.pi * sin_freq_y * t) + center_left[1]
    z_left = radius * np.sin(2 * np.pi * sin_freq_z * t) + center_left[2]
    target_pose_left.position = np.array([x_left, y_left, z_left])
    
    # Right arm figure eight with 180° phase shift (moves opposite to left arm)
    x_right = center_right[0]
    y_right = radius * np.sin(2 * np.pi * sin_freq_y * t + np.pi) + center_right[1]  # +π phase shift
    z_right = radius * np.sin(2 * np.pi * sin_freq_z * t + np.pi) + center_right[2]  # +π phase shift
    target_pose_right.position = np.array([x_right, y_right, z_right])

    left_arm.set_target(pose=target_pose_left)
    right_arm.set_target(pose=target_pose_right)

    rate.sleep()

    ee_poses_left.append(left_arm.end_effector_pose.copy())
    target_poses_left.append(left_arm._target_pose.copy())
    ee_poses_right.append(right_arm.end_effector_pose.copy())
    target_poses_right.append(right_arm._target_pose.copy())
    ts.append(t)

    t += 1.0 / ctrl_freq

while t < max_time + 1.0:
    # Just wait a bit for the end effectors to settle

    rate.sleep()

    ee_poses_left.append(left_arm.end_effector_pose.copy())
    target_poses_left.append(left_arm._target_pose.copy())
    ee_poses_right.append(right_arm.end_effector_pose.copy())
    target_poses_right.append(right_arm._target_pose.copy())
    ts.append(t)

    t += 1.0 / ctrl_freq


print("Done drawing figure eight patterns!")


# %%
# Extract positions for left arm
y_t_left = [target_pose_sample.position[1] for target_pose_sample in target_poses_left]
z_t_left = [target_pose_sample.position[2] for target_pose_sample in target_poses_left]
y_ee_left = [ee_pose.position[1] for ee_pose in ee_poses_left]
z_ee_left = [ee_pose.position[2] for ee_pose in ee_poses_left]

# Extract positions for right arm
y_t_right = [target_pose_sample.position[1] for target_pose_sample in target_poses_right]
z_t_right = [target_pose_sample.position[2] for target_pose_sample in target_poses_right]
y_ee_right = [ee_pose.position[1] for ee_pose in ee_poses_right]
z_ee_right = [ee_pose.position[2] for ee_pose in ee_poses_right]

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Left arm plots
ax[0, 0].plot(y_ee_left, z_ee_left, label="Left arm current", color='blue')
ax[0, 0].plot(y_t_left, z_t_left, label="Left arm target", linestyle="--", color='lightblue')
ax[0, 0].set_xlabel("$y$")
ax[0, 0].set_ylabel("$z$")
ax[0, 0].set_title("Left Arm Figure Eight (YZ plane)")
ax[0, 0].legend()
ax[0, 0].grid()

ax[0, 1].plot(ts, z_ee_left, label="Left arm current", color='blue')
ax[0, 1].plot(ts, z_t_left, label="Left arm target", linestyle="--", color='lightblue')
ax[0, 1].set_xlabel("$t$")
ax[0, 1].set_ylabel("$z$")
ax[0, 1].set_title("Left Arm Z vs Time")
ax[0, 1].legend()
ax[0, 1].grid()

# Right arm plots
ax[1, 0].plot(y_ee_right, z_ee_right, label="Right arm current", color='red')
ax[1, 0].plot(y_t_right, z_t_right, label="Right arm target", linestyle="--", color='pink')
ax[1, 0].set_xlabel("$y$")
ax[1, 0].set_ylabel("$z$")
ax[1, 0].set_title("Right Arm Figure Eight (YZ plane)")
ax[1, 0].legend()
ax[1, 0].grid()

ax[1, 1].plot(ts, z_ee_right, label="Right arm current", color='red')
ax[1, 1].plot(ts, z_t_right, label="Right arm target", linestyle="--", color='pink')
ax[1, 1].set_xlabel("$t$")
ax[1, 1].set_ylabel("$z$")
ax[1, 1].set_title("Right Arm Z vs Time")
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
