# MuJoCo Visualization for Mink IK Control

This document explains the MuJoCo visualization feature added to the mink control examples.

## Overview

The new `demo_mink_fr3_control_with_viz.py` script adds optional MuJoCo visualization to the mink IK control demo. This allows you to see the robot's motion in a 3D viewer window while it executes trajectories.

## Architecture

```
┌─────────────────┐
│  Mink IK Solver │  (Computes joint targets)
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│   crisp_py Robot   │  (Sends commands via ROS2)
└────────┬───────────┘
         │
         ▼
┌─────────────────────────┐
│  C++ MuJoCo Simulator   │  (Physics simulation)
│  OR Real Robot          │
└────────┬────────────────┘
         │ Publishes /joint_states
         ▼
┌─────────────────────────┐
│ Python MuJoCo Viewer    │  (Embedded in control script)
│ (Passive rendering)     │
└─────────────────────────┘
```

### Key Points

1. **C++ MuJoCo Simulator**: Used by ros2_control for full physics simulation (dynamics, contacts, etc.)
2. **Python MuJoCo Viewer**: Separate visualization running in the same process as the control script (passive rendering only)
3. **Dual MuJoCo Usage**: The C++ simulator provides physics; Python viewer provides visualization. They use the same model but serve different purposes.

## Usage

### Prerequisites

```bash
# Terminal 1: Start the FR3 stack with simulation
cd crisp_controllers_demos
ROBOT_IP=$ROBOT_IP FRANKA_FAKE_HARDWARE=true RMW=cyclone ROS_NETWORK_INTERFACE=$ROS_NETWORK_INTERFACE docker compose up launch_franka
```

### Run with Visualization

```bash
# Terminal 2: Run control script with visualization
cd crisp_py
pixi shell -e humble

# Show commanded robot (IK solution)
python examples/demo_mink_fr3_control_with_viz.py --visualize --show-mode commanded

# Show actual robot (feedback from real/simulated robot)
python examples/demo_mink_fr3_control_with_viz.py --visualize --show-mode actual

# Show actual + print tracking error (default)
python examples/demo_mink_fr3_control_with_viz.py --visualize --show-mode both

# No visualization (same as original demo)
python examples/demo_mink_fr3_control_with_viz.py
```

## Visualization Modes

### `--show-mode commanded`
- Displays the IK solution (what the controller is commanding)
- Useful for debugging IK solver behavior
- Shows the "ideal" robot motion

### `--show-mode actual`
- Displays the actual robot state from feedback
- Shows what the robot is really doing
- Includes dynamics effects (inertia, joint limits, etc.)

### `--show-mode both` (default)
- Displays actual robot state
- Prints tracking error every second:
  - Joint space error (mrad)
  - Position error (mm)
  - Orientation error

Example output:
```
Joint error: 2.34 mrad | Pos error: 0.15 mm | Ori error: 0.0012
Joint error: 1.87 mrad | Pos error: 0.12 mm | Ori error: 0.0009
```

## Implementation Details

### Threading Model
- **Main thread**: Runs MuJoCo viewer loop at 60 FPS
- **Background thread**: Runs control loop at 100 Hz
- **Thread-safe**: Uses locks to synchronize state between threads

### Visualization Update Rate
- Control: 100 Hz
- Physics (C++ sim): 1000 Hz
- Visualization: 60 FPS
- Error printing: 1 Hz

### Model Usage
- Uses the same MuJoCo model that mink uses for IK computation
- No additional model files needed
- Single robot displayed (no overlay to keep it simple)

## Comparison to C++ MuJoCo Simulator

| Feature | C++ Simulator | Python Visualizer |
|---------|---------------|-------------------|
| Purpose | Physics simulation | Visual feedback |
| Location | ros2_control hardware interface | Control script |
| Dynamics | Full physics engine | None (passive) |
| Update Rate | 1000 Hz | 60 FPS |
| Rendering | None (headless) | Interactive window |
| State Source | Computes from torques | Reads from ROS2 topics |

## Benefits

✅ **Visual debugging**: See robot motion in real-time  
✅ **Performance analysis**: Compare commanded vs actual behavior  
✅ **Trajectory verification**: Confirm motion before running on real hardware  
✅ **Simple integration**: Single script, no extra dependencies  
✅ **Optional**: Doesn't affect control performance when disabled

## Future Enhancements

Possible additions:
- [ ] Dual robot overlay (commanded + actual simultaneously)
- [ ] Trajectory visualization (path traces)
- [ ] Interactive target setting (click to move)
- [ ] Record video of visualization
- [ ] RViz integration as alternative viewer

## Troubleshooting

### Viewer doesn't open
- Check that MuJoCo Python package is installed: `pip install mujoco`
- Ensure you have display/X11 access if running in Docker

### Performance issues
- Reduce viewer FPS: Change `time.sleep(1/60)` to `time.sleep(1/30)`
- Disable visualization for production: Don't use `--visualize` flag

### Thread cleanup warnings
- Normal on exit; threads are cleaned up automatically
- Can be ignored unless causing actual issues
