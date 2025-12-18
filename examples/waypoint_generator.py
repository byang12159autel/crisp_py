import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose
from scipy.spatial.transform import Rotation, Slerp

# Define waypoints with both position and orientation
Storage = Pose(
        position=np.array([0.0, 0.09, 0.35]),
        orientation=Rotation.from_euler('xyz', [-180.0,  52.1,    90.1], degrees=True)
        )

ready = Pose(
    position=np.array([0.30, 0.0,  0.48]),
    orientation=Rotation.from_euler('xyz', [-180, -2.51149549e-05, -4.757517], degrees=True)
    )

def add_intermediate_waypoints(start_pose, end_pose, num_intermediate=2):
    """Add waypoints between start and end to avoid singularities"""
    waypoints = []
    for t in np.linspace(0, 1, num_intermediate + 2):
        pos = (1-t) * start_pose.position + t * end_pose.position
        # Also interpolate orientation with Slerp
        slerp = Slerp([0, 1], Rotation.concatenate([start_pose.orientation, end_pose.orientation]))
        ori = slerp([t])[0]
        waypoints.append(Pose(pos, ori))
    return waypoints

def plot_waypoints_3d(waypoints, arrow_length=0.05):
    """Plot waypoints in 3D with orientation frames using quiver."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = np.array([wp.position for wp in waypoints])
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'k-', linewidth=2, label='Trajectory')
    
    # Plot waypoints as spheres
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='black', s=100, marker='o', label='Waypoints')
    
    # Plot orientation frames at each waypoint
    for i, wp in enumerate(waypoints):
        # Get rotation matrix (each column is an axis direction)
        rot_matrix = wp.orientation.as_matrix()
        
        # X-axis (red)
        ax.quiver(wp.position[0], wp.position[1], wp.position[2],
                 rot_matrix[0, 0], rot_matrix[1, 0], rot_matrix[2, 0],
                 length=arrow_length, color='red', alpha=0.8, arrow_length_ratio=0.3)
        
        # Y-axis (green)
        ax.quiver(wp.position[0], wp.position[1], wp.position[2],
                 rot_matrix[0, 1], rot_matrix[1, 1], rot_matrix[2, 1],
                 length=arrow_length, color='green', alpha=0.8, arrow_length_ratio=0.3)
        
        # Z-axis (blue)
        ax.quiver(wp.position[0], wp.position[1], wp.position[2],
                 rot_matrix[0, 2], rot_matrix[1, 2], rot_matrix[2, 2],
                 length=arrow_length, color='blue', alpha=0.8, arrow_length_ratio=0.3)
    
    # Create custom legend for orientation axes
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='black', lw=2)]
    ax.legend(custom_lines, ['X-axis', 'Y-axis', 'Z-axis', 'Trajectory'], loc='upper right')
    
    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('6D Waypoint Interpolation\n(Position + Orientation)')
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                          positions[:, 1].max()-positions[:, 1].min(),
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()


results = add_intermediate_waypoints(Storage, ready)
for i in results:
    print(f"Position: {i.position}")
    print(f"Orientation (quat): {i.orientation.as_quat()}")

print("Done")

# Visualize the waypoints
plot_waypoints_3d(results)
