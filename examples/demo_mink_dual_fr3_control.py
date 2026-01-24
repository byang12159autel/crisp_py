"""Mink IK for Dual Franka FR3 control with MuJoCo visualization.

This example demonstrates using mink's inverse kinematics solver to compute
joint positions for both arms simultaneously in pure MuJoCo simulation.

Usage:
    cd crisp_py
    pixi shell -e humble
    python examples/demo_mink_dual_fr3_control.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# Path to MuJoCo dual FR3 model
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


def main():
    # 1. Setup mink IK solver using MuJoCo dual FR3 model
    print(f"Loading MuJoCo model from: {_XML}")
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
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

    # 2. Initialize simulation from home position
    print("Initializing simulation...")
    
    # Reset to home keyframe if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    else:
        mujoco.mj_forward(model, data)
    
    # FR3 dual model: q[0:7] = left arm, q[7:14] = right arm (14 total DOFs)
    model_nq = model.nq  # Should be 14 for dual FR3
    initial_q = data.qpos.copy()
    
    print(f"Left arm initial joints: {initial_q[0:7]}")
    print(f"Right arm initial joints: {initial_q[7:14]}")
    
    # Update configuration
    configuration.update(initial_q)

    # Set posture target to current configuration
    posture_task.set_target_from_configuration(configuration)

    # Get initial end-effector poses for trajectory
    left_site_id = model.site("left_attachment_site").id
    right_site_id = model.site("right_attachment_site").id
    
    left_initial_position = data.site_xpos[left_site_id].copy()
    right_initial_position = data.site_xpos[right_site_id].copy()
    
    # Get initial orientations
    left_initial_mat = data.site_xmat[left_site_id].copy().reshape(3, 3)
    right_initial_mat = data.site_xmat[right_site_id].copy().reshape(3, 3)
    
    left_initial_rotation = mink.SO3.from_matrix(left_initial_mat)
    right_initial_rotation = mink.SO3.from_matrix(right_initial_mat)
    
    print(f"Left arm initial EE position: {left_initial_position}")
    print(f"Right arm initial EE position: {right_initial_position}")

    # 3. Circular trajectory parameters
    amplitude = 0.08  # 8cm radius (smaller for dual arm safety)
    frequency = 0.15  # 0.15 Hz (~6.7 second period)

    local_time = 0.0
    rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)

    print(f"\nStarting synchronized circular trajectories")
    print(f"  Amplitude: {amplitude}m, Frequency: {frequency}Hz")
    print(f"  Left arm: counter-clockwise in XY plane")
    print(f"  Right arm: clockwise in XY plane (mirrored)")
    print("Press Ctrl+C or close viewer window to stop\n")

    # 4. Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Setup camera
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        try:
            while viewer.is_running():
                dt = rate.dt
                local_time += dt

                # 1. Update mink configuration from simulation state (closed-loop feedback)
                current_q = data.qpos.copy()
                configuration.update(current_q)

                # Debug: Print every 100 iterations to check state updates
                if local_time > 0 and int(local_time * CONTROL_FREQ) % 100 == 0:
                    print(f"[DEBUG] left_q: {current_q[0:7].round(3)}, right_q: {current_q[7:14].round(3)}")

                # 2. Compute circular offsets - mirrored for left/right arms
                phase = 2 * np.pi * frequency * local_time
                
                # Left arm: counter-clockwise
                left_offset = np.array([
                    amplitude * np.cos(phase),
                    amplitude * np.sin(phase),
                    0.0,
                ])
                
                # Right arm: clockwise (mirrored in Y)
                right_offset = np.array([
                    amplitude * np.cos(phase),
                    -amplitude * np.sin(phase),  # Mirrored
                    0.0,
                ])

                # 3. Create target SE3 poses
                left_target_position = left_initial_position + left_offset
                right_target_position = right_initial_position + right_offset
                
                left_target_SE3 = mink.SE3.from_rotation_and_translation(
                    left_initial_rotation, left_target_position
                )
                right_target_SE3 = mink.SE3.from_rotation_and_translation(
                    right_initial_rotation, right_target_position
                )

                # 4. Set IK targets and solve for both arms simultaneously
                left_ee_task.set_target(left_target_SE3)
                right_ee_task.set_target(right_target_SE3)
                converge_ik(configuration, tasks, dt, SOLVER, MAX_ITERS)

                # 5. Extract joint positions and send to simulation
                q_target = configuration.q[:14]  # Full commanded configuration
                
                # Set controls (FR3 dual: ctrl[0:7] = left arm, ctrl[7:14] = right arm)
                data.ctrl[0:7] = q_target[0:7]
                data.ctrl[7:14] = q_target[7:14]

                # 6. Step simulation
                mujoco.mj_step(model, data)
                
                # 7. Update viewer
                viewer.sync()
                
                # Compute and report tracking errors periodically
                if int(local_time * CONTROL_FREQ) % 100 == 0:
                    joint_error = np.linalg.norm(q_target - current_q)
                    
                    # Compute position/orientation errors for both end-effectors
                    left_err = left_ee_task.compute_error(configuration)
                    right_err = right_ee_task.compute_error(configuration)
                    
                    left_pos_error = np.linalg.norm(left_err[:3])
                    left_ori_error = np.linalg.norm(left_err[3:])
                    right_pos_error = np.linalg.norm(right_err[:3])
                    right_ori_error = np.linalg.norm(right_err[3:])
                    
                    print(f"[t={local_time:.2f}s] Joint error: {joint_error:.6f}, "
                          f"Left pos/ori: {left_pos_error:.6f}/{left_ori_error:.6f}, "
                          f"Right pos/ori: {right_pos_error:.6f}/{right_ori_error:.6f}")

                rate.sleep()

        except KeyboardInterrupt:
            print("\nStopping...")

    print("Simulation complete.")


if __name__ == "__main__":
    main()
