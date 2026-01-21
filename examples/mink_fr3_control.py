"""Mink IK with crisp_py for Franka FR3 control.

This example demonstrates using mink's inverse kinematics solver to compute
joint positions, which are then sent to a real/simulated FR3 robot via crisp_py.

Usage:
    1. Start the FR3 stack (simulation):
       cd crisp_controllers_demos
       ROBOT_IP=$ROBOT_IP FRANKA_FAKE_HARDWARE=true RMW=cyclone ROS_NETWORK_INTERFACE=$ROS_NETWORK_INTERFACE docker compose up launch_franka

    2. Run this script:
       cd crisp_py
       pixi shell -e humble
       python examples/mink_fr3_control.py
"""

from pathlib import Path

import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from crisp_py.robot import make_robot

# Path to MuJoCo model (using Panda model, kinematically similar to FR3)
_HERE = Path(__file__).parent.parent / "mink" / "examples"
_XML = _HERE / "franka_emika_panda" / "mjx_scene.xml"

# IK configuration
SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20
CONTROL_FREQ = 100.0  # Hz


def converge_ik(configuration, tasks, dt, solver, max_iters):
    """Run IK iterations until convergence or max_iters reached."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)

        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= POS_THRESHOLD
        ori_achieved = np.linalg.norm(err[3:]) <= ORI_THRESHOLD

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    # 1. Setup mink IK solver using MuJoCo model
    print(f"Loading MuJoCo model from: {_XML}")
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    # Create end-effector task for IK
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    # 2. Setup crisp_py robot interface
    print("Connecting to FR3 via crisp_py...")
    robot = make_robot("fr3")
    robot.wait_until_ready()
    print("Robot ready!")

    # Switch to joint impedance controller for position control
    robot.controller_switcher_client.switch_controller("joint_impedance_controller")
    print("Switched to joint_impedance_controller")

    # 3. Initialize mink configuration from real robot state
    # Note: FR3 has 7 joints, model may have 8 (with gripper)
    initial_q = robot.joint_values
    print(f"Initial joint configuration: {initial_q}")

    # Pad to model size if needed (gripper DOF)
    model_nq = model.nq
    if len(initial_q) < model_nq:
        padded_q = np.zeros(model_nq)
        padded_q[: len(initial_q)] = initial_q
        configuration.update(padded_q)
    else:
        configuration.update(initial_q[:model_nq])

    # Set posture target to current configuration
    posture_task.set_target_from_configuration(configuration)

    # Get initial end-effector pose for trajectory
    initial_pose = robot.end_effector_pose
    initial_position = initial_pose.position
    initial_rotation = mink.SO3.from_matrix(initial_pose.orientation.as_matrix())
    print(f"Initial end-effector position: {initial_position}")

    # 4. Circular trajectory parameters
    amplitude = 0.10  # 10cm radius
    frequency = 0.2  # 0.2 Hz (5 second period)

    local_time = 0.0
    rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)

    print(f"\nStarting circular trajectory (amplitude={amplitude}m, freq={frequency}Hz)")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            dt = rate.dt
            local_time += dt

            # Compute circular offset in XY plane
            offset = np.array(
                [
                    amplitude * np.cos(2 * np.pi * frequency * local_time),
                    amplitude * np.sin(2 * np.pi * frequency * local_time),
                    0.0,
                ]
            )

            # Create target SE3 pose
            target_position = initial_position + offset
            target_SE3 = mink.SE3.from_rotation_and_translation(
                initial_rotation, target_position
            )

            # Set IK target and solve
            end_effector_task.set_target(target_SE3)
            converge_ik(configuration, tasks, dt, SOLVER, MAX_ITERS)

            # Send joint positions to robot (only first 7 for FR3)
            q_target = configuration.q[:7]
            robot.set_target_joint(q_target)

            rate.sleep()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.shutdown()
        print("Robot shutdown complete.")


if __name__ == "__main__":
    main()
