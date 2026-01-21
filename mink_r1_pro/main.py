
# =====================================================================
# 第一步：启动 Isaac Sim（必须最先执行）
# =====================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="R1 Pro IK + Swerve Control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =====================================================================
# 第二步：导入其他模块（仿真器启动后）
# =====================================================================
import os
import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from isaaclab.envs import ManagerBasedEnv
# 统一使用 PoseUtils 进行姿态变换（符合 Isaac Lab Mimic 代码风格）
import isaaclab.utils.math as PoseUtils

from r1pro_ik_wbc import R1ProIKController
import r1pro_cfg

# 用于创建可拖动的目标球体（不通过 Isaac Lab 管理）
from pxr import UsdGeom, Gf


# =====================================================================
# Helper 函数：姿态变换（符合 Isaac Lab PoseUtils 风格）
# =====================================================================
def get_usd_prim_pose_w(prim, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 USD Prim 获取世界坐标系下的位姿 (pos, quat)。
    
    Args:
        prim: USD Prim 对象
        device: torch 设备
    
    Returns:
        pos_w: (1, 3) 世界坐标位置
        quat_w: (1, 4) 世界坐标四元数 [w, x, y, z]
    """
    xformable = UsdGeom.Xformable(prim)
    world_xform = xformable.ComputeLocalToWorldTransform(0)
    # 提取位置
    pos = world_xform.GetRow(3)[:3]
    pos_w = torch.tensor([[pos[0], pos[1], pos[2]]], device=device)
    # 提取四元数
    xform = Gf.Transform()
    xform.SetMatrix(world_xform)
    quat = xform.GetRotation().GetQuat()
    quat_w = torch.tensor([[quat.GetReal(), quat.GetImaginary()[0], 
                            quat.GetImaginary()[1], quat.GetImaginary()[2]]], device=device)
    return pos_w, quat_w


def get_target_pose_in_base_frame(robot, target_prim, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    获取目标物体在机器人基座坐标系下的位姿。
    模仿 Isaac Lab Mimic 中 get_object_poses 的实现。
    
    Args:
        robot: Isaac Lab Articulation 对象
        target_prim: 目标 USD Prim
        device: torch 设备
    
    Returns:
        pos_b: (1, 3) 基座坐标系下的位置
        quat_b: (1, 4) 基座坐标系下的四元数
    """
    # 获取目标世界姿态
    pos_w, quat_w = get_usd_prim_pose_w(target_prim, device)
    # 转换到机器人基座坐标系
    pos_b, quat_b = PoseUtils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w
    )
    return pos_b, quat_b


def get_eef_pose_in_base_frame(
    robot, 
    body_idx: int, 
    offset_pos: torch.Tensor | None = None, 
    offset_rot: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    获取末端执行器在机器人基座坐标系下的位姿（可选应用 offset）。
    
    Args:
        robot: Isaac Lab Articulation 对象
        body_idx: body 索引
        offset_pos: (1, 3) 可选的位置偏移
        offset_rot: (1, 4) 可选的旋转偏移 [w, x, y, z]
    
    Returns:
        pos_b: (1, 3) 基座坐标系下的位置
        quat_b: (1, 4) 基座坐标系下的四元数
    """
    # 获取 body 世界姿态
    body_pos_w = robot.data.body_pos_w[:, body_idx]
    body_quat_w = robot.data.body_quat_w[:, body_idx]
    # 转换到基座坐标系
    pos_b, quat_b = PoseUtils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, body_pos_w, body_quat_w
    )
    # 应用 offset
    if offset_pos is not None and offset_rot is not None:
        pos_b, quat_b = PoseUtils.combine_frame_transforms(pos_b, quat_b, offset_pos, offset_rot)
    return pos_b, quat_b


# =====================================================================
# R1ProBridge - ROS 2 桥接节点
# =====================================================================
class R1ProBridge(Node):
    """连接 Isaac Lab 和 ROS 2 Swerve Controller"""
    
    STEER_JOINTS = ["steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3"]
    WHEEL_JOINTS = ["wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3"]
    
    # IK 输出顺序
    IK_JOINT_ORDER = [
        "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
        "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
        "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
        "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
    ]
    
    # Isaac Lab action 顺序
    ACTION_JOINT_ORDER = [
        "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
        "left_arm_joint1", "right_arm_joint1",
        "left_arm_joint2", "right_arm_joint2",
        "left_arm_joint3", "right_arm_joint3",
        "left_arm_joint4", "right_arm_joint4",
        "left_arm_joint5", "right_arm_joint5",
        "left_arm_joint6", "right_arm_joint6",
        "left_arm_joint7", "right_arm_joint7",
    ]
    
    def __init__(self, robot):
        super().__init__("r1_pro_ik_bridge")
        self.robot = robot
        self.device = robot.device
        
        all_joints = list(robot.data.joint_names)
        self.steer_idxs = [all_joints.index(n) for n in self.STEER_JOINTS]
        self.wheel_idxs = [all_joints.index(n) for n in self.WHEEL_JOINTS]
        self.upper_idxs = [all_joints.index(n) for n in self.IK_JOINT_ORDER]
        self.ik_to_action_map = [self.ACTION_JOINT_ORDER.index(n) for n in self.IK_JOINT_ORDER]
        
        self.wheel_state_pub = self.create_publisher(JointState, "/Swerve_wheel_jointstates", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        self.swerve_cmd = None
        self.create_subscription(JointState, "/Swerve_wheel_jointcommand", self._swerve_callback, 10)
    
    def _swerve_callback(self, msg: JointState):
        self.swerve_cmd = msg
    
    def publish_wheel_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.STEER_JOINTS + self.WHEEL_JOINTS
        
        joint_pos = self.robot.data.joint_pos[0].cpu().numpy()
        joint_vel = self.robot.data.joint_vel[0].cpu().numpy()
        
        msg.position = [joint_pos[i] for i in self.steer_idxs] + [joint_pos[i] for i in self.wheel_idxs]
        msg.velocity = [joint_vel[i] for i in self.steer_idxs] + [joint_vel[i] for i in self.wheel_idxs]
        self.wheel_state_pub.publish(msg)
    
    def publish_cmd_vel(self, base_velocity: np.ndarray):
        msg = Twist()
        msg.linear.x = float(base_velocity[0])
        msg.linear.y = float(base_velocity[1])
        msg.angular.z = float(base_velocity[2])
        self.cmd_vel_pub.publish(msg)
    
    def get_swerve_command(self) -> tuple:
        if self.swerve_cmd is None:
            return np.zeros(3), np.zeros(3)
        
        msg = self.swerve_cmd
        steer_cmd = np.zeros(3)
        wheel_cmd = np.zeros(3)
        
        for i, name in enumerate(msg.name):
            if name in self.STEER_JOINTS:
                idx = self.STEER_JOINTS.index(name)
                steer_cmd[idx] = msg.position[i] if i < len(msg.position) else 0.0
            elif name in self.WHEEL_JOINTS:
                idx = self.WHEEL_JOINTS.index(name)
                wheel_cmd[idx] = msg.velocity[i] if i < len(msg.velocity) else 0.0
        
        return steer_cmd, wheel_cmd
    
    def get_upper_body_joints(self) -> np.ndarray:
        joint_pos = self.robot.data.joint_pos[0].cpu().numpy()
        return np.array([joint_pos[i] for i in self.upper_idxs])


# =====================================================================
# 主函数
# =====================================================================
def main():
    # 创建环境
    env_cfg = r1pro_cfg.R1ProEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    
    # 获取柠檬目标球体（在 r1pro_cfg.py 中通过 AssetBaseCfg 定义）
    # 路径格式：{ENV_REGEX_NS}/lemon -> /World/envs/env_0/lemon
    lemon_prim_path = "/World/envs/env_0/lemon"
    stage = env.sim.stage
    lemon_prim = stage.GetPrimAtPath(lemon_prim_path)
    print(f"[Main] 柠檬目标: {lemon_prim_path}")
    
    # 末端位置：使用 link7 + offset（补偿 MuJoCo geom 和 Isaac link frame 差异）
    body_names = list(robot.data.body_names)
    left_gripper_idx = body_names.index("left_arm_link7")
    right_gripper_idx = body_names.index("right_arm_link7")
    # offset: URDF joint origin（与 MuJoCo geom pos 一致）
    GRIPPER_OFFSET_POS = torch.tensor([[-0.0295, 0.0, -0.16065]], device=env.device)
    GRIPPER_OFFSET_ROT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    print(f"[Main] 使用 link7 + offset: left={left_gripper_idx}, right={right_gripper_idx}")
    
    # 初始化 IK 控制器
    xml_path = os.path.join(os.path.dirname(__file__), "r1_pro_ik_scene.xml")
    ik_controller = R1ProIKController(xml_path)
    ik_controller.initialize()
    print("[Main] IK 控制器初始化完成")
    
    # 初始化 ROS 2
    rclpy.init()
    bridge = R1ProBridge(robot)
    print("[Main] ROS 2 桥接节点初始化完成")
    
    # 重置环境
    obs, _ = env.reset()
    
    # 右手目标（保持初始位置）
    initial_joints = bridge.get_upper_body_joints()
    _, right_target = ik_controller.get_current_hand_poses(initial_joints)
    
    print(f"[Main] 右手目标: {right_target[:3]}")
    print("[Main] 开始主循环...")
    
    # 控制频率 = sim_dt * decimation（符合 Isaac Lab ManagerBasedEnv 风格）
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    
    # =====================================================================
    # 主循环：Perception -> Planning -> Actuation -> Logging
    # =====================================================================
    while simulation_app.is_running():
        loop_start = time.time()
        
        with torch.inference_mode():
            # -----------------------------------------------------------
            # 1. 感知 (Perception)
            # -----------------------------------------------------------
            rclpy.spin_once(bridge, timeout_sec=0.0)
            bridge.publish_wheel_states()
            
            # 获取目标位姿（机器人基座坐标系）
            target_pos_b, target_quat_b = get_target_pose_in_base_frame(robot, lemon_prim, env.device)
            left_target = torch.cat([target_pos_b[0], target_quat_b[0]]).cpu().numpy()
            
            # 获取当前关节状态
            current_upper = bridge.get_upper_body_joints()
            
            # -----------------------------------------------------------
            # 2. 规划 (Planning) - IK 求解
            # -----------------------------------------------------------
            result = ik_controller.solve(current_upper, left_target, right_target, control_dt)
            
            # -----------------------------------------------------------
            # 3. 执行 (Actuation)
            # -----------------------------------------------------------
            # 发布 cmd_vel 给 Swerve Controller
            bridge.publish_cmd_vel(result.base_velocity)
            
            # 获取轮子命令
            steer_cmd, wheel_cmd = bridge.get_swerve_command()
            
            # 组装 Action Tensor (28维: upper_body 18 + steer 3 + wheel 3 + gripper 4)
            action = torch.zeros((1, 28), device=env.device)
            
            # 重映射 IK 输出到 Action 顺序
            ik_positions = result.joint_positions
            action_positions = np.zeros(18)
            for ik_idx, action_idx in enumerate(bridge.ik_to_action_map):
                action_positions[action_idx] = ik_positions[ik_idx]
            
            action[0, :18] = torch.from_numpy(action_positions).float()
            action[0, 18:21] = torch.from_numpy(steer_cmd).float()
            action[0, 21:24] = torch.from_numpy(wheel_cmd).float()
            # action[0, 24:28] 夹爪保持默认值 0（张开状态）
            
            # 环境步进
            obs, _ = env.step(action)
            
        
        # -----------------------------------------------------------
        # 4. 时间同步
        # -----------------------------------------------------------
        elapsed = time.time() - loop_start
        if elapsed < control_dt:
            time.sleep(control_dt - elapsed)
    
    bridge.destroy_node()
    rclpy.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
