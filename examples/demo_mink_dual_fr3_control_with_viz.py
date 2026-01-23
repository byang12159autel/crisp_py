"""Mink IK with crisp_py for Franka FR3 control with optional MuJoCo visualization.

This example demonstrates using mink's inverse kinematics solver to compute
joint positions, which are then sent to a real/simulated FR3 robot via crisp_py.
Optionally visualizes the robot motion using MuJoCo's passive viewer.

Usage:
    1. Start the FR3 stack (simulation):
       cd crisp_controllers_demos
       ROBOT_IP=$ROBOT_IP FRANKA_FAKE_HARDWARE=true RMW=cyclone ROS_NETWORK_INTERFACE=$ROS_NETWORK_INTERFACE docker compose up launch_franka

    2. Run this script with visualization:
       cd crisp_py
       pixi shell -e humble
       
       # Show commanded robot (IK solution)
       python examples/demo_mink_fr3_control_with_viz.py --visualize --show-mode commanded
       
       # Show actual robot (real state from feedback)
       python examples/demo_mink_fr3_control_with_viz.py --visualize --show-mode actual
       
       # Show actual + print tracking error (default)
       python examples/demo_mink_fr3_control_with_viz.py --visualize --show-mode both
       
       # No visualization
       python examples/demo_mink_fr3_control_with_viz.py
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from crisp_py.robot import make_robot
from crisp_py.utils import MuJoCoVisualizer

from crisp_py.robot import make_robot

from crisp_py.robot import Robot, FrankaConfig

# Path to MuJoCo model (dual FR3 model)
_XML = Path("/home/ben/crisp_framework/crisp_controllers_demos/crisp_controllers_robot_demos/config/fr3/scene_dual.xml")

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

        # Check convergence for both end-effector tasks
        left_err = tasks[0].compute_error(configuration)
        right_err = tasks[1].compute_error(configuration)
        
        left_pos_achieved = np.linalg.norm(left_err[:3]) <= POS_THRESHOLD
        left_ori_achieved = np.linalg.norm(left_err[3:]) <= ORI_THRESHOLD
        right_pos_achieved = np.linalg.norm(right_err[:3]) <= POS_THRESHOLD
        right_ori_achieved = np.linalg.norm(right_err[3:]) <= ORI_THRESHOLD

        if left_pos_achieved and left_ori_achieved and right_pos_achieved and right_ori_achieved:
            return True
    return False

def pad_joints(joint_array, target_size):
    """Pad joint array to target size with zeros."""
    if len(joint_array) < target_size:
        padded = np.zeros(target_size)
        padded[: len(joint_array)] = joint_array
        return padded
    return joint_array[:target_size]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Mink IK control with optional MuJoCo visualization')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable MuJoCo visualization')
    parser.add_argument('--show-mode', type=str, 
                       choices=['commanded', 'actual', 'both'],
                       default='both',
                       help='Visualization mode: commanded (IK solution), actual (robot feedback), both (actual + error print)')
    args = parser.parse_args()
    
    # 1. Setup mink IK solver using MuJoCo model
    print(f"Loading MuJoCo model from: {_XML}")
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)
    
    # Create end-effector tasks for both arms
    left_ee_task = mink.FrameTask(
        frame_name="left_attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    right_ee_task = mink.FrameTask(
        frame_name="right_attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [left_ee_task, right_ee_task, posture_task]
    
    # 2. Setup crisp_py robot interface
    print("Connecting to FR3 via crisp_py...")
    faster_publishing_config = FrankaConfig()
    faster_publishing_config.publish_frequency = 20.0
    faster_publishing_config.use_prefix = True  # Required for dual arm with namespaces
    left_arm = Robot(robot_config=faster_publishing_config, namespace="left")
    right_arm = Robot(robot_config=faster_publishing_config, namespace="right")

    left_arm.wait_until_ready()
    right_arm.wait_until_ready()
    print("Robot ready!")
    
    # Switch to joint impedance controller for position control
    left_arm.controller_switcher_client.switch_controller("joint_impedance_controller")
    right_arm.controller_switcher_client.switch_controller("joint_impedance_controller")
    print("Switched to joint_impedance_controller")
    
    # 3. Initialize mink configuration from real robot state
    # Concatenate both arms: q[0:7] = left arm, q[7:14] = right arm
    initial_q = np.concatenate([left_arm.joint_values, right_arm.joint_values])
    print(f"Initial joint configuration (14 DOFs):")
    print(f"  Left arm:  {initial_q[0:7]}")
    print(f"  Right arm: {initial_q[7:14]}")
    
    # Pad to model size if needed (gripper DOF)
    model_nq = model.nq
    padded_q = pad_joints(initial_q, model_nq)
    configuration.update(padded_q)
    
    # Set posture target to current configuration
    posture_task.set_target_from_configuration(configuration)
    
    # Get initial end-effector poses for both arms
    left_initial_pose = left_arm.end_effector_pose
    left_initial_position = left_initial_pose.position
    left_initial_rotation = mink.SO3.from_matrix(left_initial_pose.orientation.as_matrix())
    
    right_initial_pose = right_arm.end_effector_pose
    right_initial_position = right_initial_pose.position
    right_initial_rotation = mink.SO3.from_matrix(right_initial_pose.orientation.as_matrix())
    
    print(f"Initial end-effector positions:")
    print(f"  Left arm:  {left_initial_position}")
    print(f"  Right arm: {right_initial_position}")
    
    # 4. Setup visualization if requested
    visualizer = None
    if args.visualize:
        print(f"\nStarting visualization (mode: {args.show_mode})...")
        visualizer = MuJoCoVisualizer(model, padded_q, mode=args.show_mode)
        visualizer.set_control_frequency(CONTROL_FREQ)
        visualizer.start()
    
    # 5. Circular trajectory parameters
    amplitude = 0.10  # 10cm radius
    frequency = 0.2  # 0.2 Hz (5 second period)
    
    local_time = 0.0
    rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)
    
    print(f"\nStarting circular trajectory (amplitude={amplitude}m, freq={frequency}Hz)")
    print("Press Ctrl+C or close viewer window to stop\n")
    
    try:
        while True:
            # Check if visualizer is still running (if enabled)
            if visualizer is not None and not visualizer.is_running():
                print("Viewer window closed")
                break
                
            dt = rate.dt
            local_time += dt
            
            # 1. Update mink configuration from actual robot joint positions (closed-loop feedback)
            current_q = np.concatenate([left_arm.joint_values, right_arm.joint_values])
            padded_q = pad_joints(current_q, model_nq)
            configuration.update(padded_q)
            
            # 2. Compute circular offsets for both arms
            phase = 2 * np.pi * frequency * local_time
            
            # Left arm: counter-clockwise in XY plane
            left_offset = np.array([
                amplitude * np.cos(phase),
                amplitude * np.sin(phase),
                0.0,
            ])
            
            # Right arm: clockwise (mirrored in Y)
            right_offset = np.array([
                amplitude * np.cos(phase),
                -amplitude * np.sin(phase),
                0.0,
            ])
            
            # 3. Create target SE3 poses for both arms
            left_target_position = left_initial_position + left_offset
            right_target_position = right_initial_position + right_offset
            
            left_target_SE3 = mink.SE3.from_rotation_and_translation(
                left_initial_rotation, left_target_position
            )
            right_target_SE3 = mink.SE3.from_rotation_and_translation(
                right_initial_rotation, right_target_position
            )
            
            # 4. Set IK targets for both arms and solve
            left_ee_task.set_target(left_target_SE3)
            right_ee_task.set_target(right_target_SE3)
            converge_ik(configuration, tasks, dt, SOLVER, MAX_ITERS)
            
            # 5. Send joint positions to both arms (14 DOFs total)
            q_target = configuration.q[:14]  # Left arm [0:7] + Right arm [7:14]
            left_arm.set_target_joint(q_target[0:7])
            right_arm.set_target_joint(q_target[7:14])
            
            # 6. Update visualization if enabled
            if visualizer is not None:
                visualizer.update_commanded(q_target)
                visualizer.update_actual(current_q)
                
                # Compute and report tracking errors for 'both' mode
                if args.show_mode == 'both':
                    joint_error = np.linalg.norm(q_target - current_q)
                    
                    # Compute position/orientation errors for both end-effectors
                    left_err = left_ee_task.compute_error(configuration)
                    right_err = right_ee_task.compute_error(configuration)
                    
                    left_pos_error = np.linalg.norm(left_err[:3])
                    left_ori_error = np.linalg.norm(left_err[3:])
                    right_pos_error = np.linalg.norm(right_err[:3])
                    right_ori_error = np.linalg.norm(right_err[3:])
                    
                    # Use average errors for visualization
                    avg_pos_error = (left_pos_error + right_pos_error) / 2
                    avg_ori_error = (left_ori_error + right_ori_error) / 2
                    visualizer.set_error_metrics(joint_error, avg_pos_error, avg_ori_error)
            
            rate.sleep()
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    finally:
        # 7. Cleanup
        if visualizer is not None:
            visualizer.stop()
        
        print("Shutting down both arms...")
        left_arm.shutdown()
        right_arm.shutdown()
        print("Robot shutdown complete.")


if __name__ == "__main__":
    main()
