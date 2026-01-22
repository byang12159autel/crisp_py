"""MuJoCo visualization utility for robot control loops.

This module provides a reusable, thread-safe MuJoCo visualization class
that can be easily integrated into robot control examples and applications.
"""

import threading
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np


class MuJoCoVisualizer:
    """Thread-safe MuJoCo visualization for robot control loops.
    
    This class manages a MuJoCo viewer in a separate thread, allowing you to
    visualize robot motion while running control loops. It supports multiple
    visualization modes:
    
    - 'commanded': Shows the commanded/desired robot state (IK solution)
    - 'actual': Shows the actual robot state from feedback
    - 'both': Shows actual state and prints tracking errors
    
    Example:
        >>> model = mujoco.MjModel.from_xml_path("robot.xml")
        >>> visualizer = MuJoCoVisualizer(model, initial_qpos, mode='actual')
        >>> visualizer.start()
        >>> 
        >>> # In control loop:
        >>> while visualizer.is_running():
        >>>     visualizer.update_actual(robot.joint_values)
        >>>     # ... control logic ...
        >>> 
        >>> visualizer.stop()
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        initial_qpos: np.ndarray,
        mode: str = 'actual',
        error_print_freq: float = 1.0
    ):
        """Initialize the MuJoCo visualizer.
        
        Args:
            model: MuJoCo model (MjModel) to visualize
            initial_qpos: Initial joint positions for the model
            mode: Visualization mode ('commanded', 'actual', or 'both')
            error_print_freq: Frequency (Hz) to print tracking errors in 'both' mode
        """
        if mode not in ['commanded', 'actual', 'both']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'commanded', 'actual', or 'both'")
        
        self.model = model
        self.mode = mode
        self.error_print_freq = error_print_freq
        
        # Thread-safe shared state
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._viewer = None
        self._viewer_thread = None
        
        # MuJoCo data
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial_qpos
        
        # State tracking
        self._commanded_q = initial_qpos.copy()
        self._actual_q = initial_qpos.copy()
        self._loop_count = 0
        
    def update_commanded(self, q: np.ndarray):
        """Update the commanded joint positions.
        
        Args:
            q: Commanded joint position array
        """
        with self._lock:
            self._commanded_q[:len(q)] = q
            if self.mode == 'commanded':
                self.data.qpos[:len(q)] = q
                
    def update_actual(self, q: np.ndarray):
        """Update the actual joint positions from robot feedback.
        
        Args:
            q: Actual joint position array
        """
        with self._lock:
            self._actual_q[:len(q)] = q
            if self.mode in ['actual', 'both']:
                self.data.qpos[:len(q)] = q
                
    def set_error_metrics(self, joint_error: float, pos_error: float, ori_error: float):
        """Set tracking error metrics for printing in 'both' mode.
        
        Args:
            joint_error: Joint space error (rad)
            pos_error: Position error (m)
            ori_error: Orientation error (rad)
        """
        if self.mode == 'both':
            with self._lock:
                self._loop_count += 1
                # Print at specified frequency
                if hasattr(self, '_control_freq'):
                    print_interval = int(self._control_freq / self.error_print_freq)
                    if self._loop_count % print_interval == 0:
                        print(f"Joint error: {joint_error*1000:.2f} mrad | "
                              f"Pos error: {pos_error*1000:.2f} mm | "
                              f"Ori error: {ori_error:.4f}")
    
    def set_control_frequency(self, freq: float):
        """Set control loop frequency for error printing calculations.
        
        Args:
            freq: Control frequency in Hz
        """
        self._control_freq = freq
        
    def start(self):
        """Launch the MuJoCo viewer in a separate thread."""
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            print("Visualizer already running")
            return
            
        self._stop_event.clear()
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()
        
        # Give viewer time to initialize
        time.sleep(0.5)
        
        print(f"MuJoCo viewer launched (mode: {self.mode})")
        if self.mode == 'commanded':
            print("  → Showing commanded robot (IK solution)")
        elif self.mode == 'actual':
            print("  → Showing actual robot (feedback state)")
        else:
            print("  → Showing actual robot + printing tracking error")
            
    def stop(self):
        """Stop the visualization and cleanup resources."""
        if self._viewer_thread is None:
            return
            
        print("\nStopping visualizer...")
        self._stop_event.set()
        
        # Wait for thread to finish
        self._viewer_thread.join(timeout=2.0)
        if self._viewer_thread.is_alive():
            print("Warning: Viewer thread did not stop cleanly")
        
        self._viewer = None
        self._viewer_thread = None
        print("Visualizer stopped")
        
    def is_running(self) -> bool:
        """Check if the viewer is still active.
        
        Returns:
            True if viewer window is open and running, False otherwise
        """
        if self._viewer is None:
            return False
        return self._viewer.is_running() and not self._stop_event.is_set()
        
    def _viewer_loop(self):
        """Internal viewer loop running in separate thread."""
        try:
            # Launch passive viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Run at ~60 FPS
            while self._viewer.is_running() and not self._stop_event.is_set():
                with self._lock:
                    # Forward dynamics for visualization
                    mujoco.mj_forward(self.model, self.data)
                
                # Sync viewer
                self._viewer.sync()
                time.sleep(1/60)
                
        except Exception as e:
            print(f"Error in viewer loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Signal that viewer has stopped
            self._stop_event.set()
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
