"""Simple joint control for Dual Franka FR3 in MuJoCo simulation.

This example demonstrates simple direct joint position control without IK.
Moves the base joint (joint1) of both arms to 90 degrees.

Usage:
    cd crisp_py
    pixi shell -e humble
    python examples/demo_simple_dual_fr3_control.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

# Path to MuJoCo dual FR3 model
_XML = Path("/home/ben/crisp_framework/crisp_controllers_demos/crisp_controllers_robot_demos/config/fr3/scene_dual.xml")

# Control configuration
CONTROL_FREQ = 100.0  # Hz


def main():
    # 1. Load MuJoCo model
    print(f"Loading MuJoCo model from: {_XML}")
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # 2. Initialize from home keyframe
    print("Initializing simulation from home position...")
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    else:
        mujoco.mj_forward(model, data)
    
    initial_q = data.qpos.copy()
    print(f"Initial configuration:")
    print(f"  Left arm:  {initial_q[0:7]}")
    print(f"  Right arm: {initial_q[7:14]}")
    
    # 3. Set target joint positions
    # Move base joint (joint1) to 90 degrees (π/2 radians) for both arms
    target_q = initial_q.copy()
    target_q[0] = -np.pi / 2  # Left arm joint1 -> 90 degrees
    target_q[1] = 1.7
    target_q[7] = np.pi / 2  # Right arm joint1 -> 90 degrees
    target_q[8] = 1.7
    
    print(f"\nTarget configuration:")
    print(f"  Left arm:  {target_q[0:7]}")
    print(f"  Right arm: {target_q[7:14]}")
    print(f"\nMoving base joints to 90 degrees...")
    
    # 4. Launch MuJoCo viewer and run control loop
    rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)
    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Setup camera
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        print("Viewer launched. Press Ctrl+C or close window to stop.\n")
        
        try:
            while viewer.is_running():
                # Set control targets directly
                data.ctrl[:] = target_q
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Update viewer
                viewer.sync()
                
                # Sleep to maintain control rate
                rate.sleep()
                
        except KeyboardInterrupt:
            print("\nStopping...")
    
    print("Simulation complete.")


if __name__ == "__main__":
    main()
