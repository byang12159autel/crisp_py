"""Interactive gravity compensation mode with keypress to print joint info and EE pose."""

# %%
from crisp_py.robot import make_robot
import time

robot = make_robot("fr3")
robot.wait_until_ready()
robot.home()
time.sleep(1.0)
# %%
robot.cartesian_controller_parameters_client.load_param_config(
    file_path="config/control/no_friction_cartesian_impedance.yaml"
)
robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

# %%
print("Gravity compensation mode activated.")
print("Press 'p' to print joint positions and end-effector pose.")
print("Press 'q' to quit.")

try:
    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == 'p':
            joint_positions = robot.joint_values
            ee_pose = robot.end_effector_pose

            print("\n--- Current Robot State ---")
            print(f"Joint Positions: {joint_positions}")
            print(f"End-Effector Position: {ee_pose.position}")
            print(f"End-Effector Orientation (quaternion): {ee_pose.orientation.as_quat()}")
            print(f"End-Effector Orientation (Euler XYZ degrees): {ee_pose.orientation.as_euler('xyz', degrees=True)}")
            print("---------------------------\n")
        elif cmd == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid command. Use 'p' to print or 'q' to quit.")

finally:
    robot.shutdown()
