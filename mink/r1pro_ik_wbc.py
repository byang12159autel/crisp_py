"""
R1 Pro IK 控制器

纯运动学 IK 控制器，用于计算躯干+双臂的关节位置和底盘速度。

使用方式：
    controller = R1ProIKController("path/to/model.xml")
    controller.initialize()
    result = controller.solve(left_target, right_target, dt=0.01)
    
    # result.joint_positions: 躯干+双臂关节位置 (18,)
    #   - torso_joint1-4 (4个)
    #   - left_arm_joint1-7 (7个)
    #   - right_arm_joint1-7 (7个)
    # result.base_velocity: 底盘速度 [vx, vy, wz]

目标坐标系：相对于 base_link
注意：轮子由 Swerve 控制器单独处理，不在 IK 输出中
"""

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

import mink


@dataclass
class IKResult:
    """IK 求解结果"""
    joint_positions: np.ndarray    # 躯干+双臂关节位置 (18,)
    base_velocity: np.ndarray      # 底盘速度 [vx, vy, wz]
    success: bool                  # 是否成功


class R1ProIKController:
    """
    R1 Pro 逆运动学控制器
    
    输入：当前关节状态 + 左右手目标位姿（相对于 base_link）
    输出：关节速度 + 底盘速度
    """
    
    # 关节索引定义
    N_BASE = 3       # 底盘: x, y, yaw
    N_WHEEL = 6      # 轮子: steer1-3, wheel1-3
    IDX_TORSO = 9    # 躯干起始索引
    N_TORSO = 4      # 躯干关节数
    N_ARM = 7        # 单臂关节数
    N_ARM_TOTAL = 18 # 躯干(4) + 左臂(7) + 右臂(7)
    
    def __init__(self, xml_path: str, solver: str = "daqp"):
        """
        初始化 IK 控制器
        
        Args:
            xml_path: MuJoCo XML 模型路径
            solver: QP 求解器（默认 daqp）
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.configuration = mink.Configuration(self.model)
        self.solver = solver
        
        # 更新引用（mink 可能会修改）
        self.model = self.configuration.model
        self.data = self.configuration.data
        
        # 从模型获取实际自由度数量
        self.nv = self.model.nv
        self.nq = self.model.nq
        print(f"模型加载完成: nq={self.nq}, nv={self.nv}")
        
        # ===== 定义 IK 任务 =====
        # 姿态保持成本：底盘低成本，躯干手臂高成本
        posture_cost = np.zeros((self.model.nv,))
        posture_cost[:self.N_BASE] = 2.0      # 底盘：倾向保持不动
        posture_cost[self.N_BASE:] = 1e-1     # 躯干+手臂
        
        # 躯干姿态任务（保持直立）
        self.pelvis_task = mink.FrameTask(
            frame_name="torso_link4",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=2.0,
            lm_damping=1.0,
        )
        self.torso_task = mink.FrameTask(
            frame_name="torso_link2",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=3.0,
            lm_damping=1.0,
        )
        
        # 姿态任务
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        
        # 左右手任务
        self.left_hand_task = mink.FrameTask(
            frame_name="left_gripper_link",
            frame_type="geom",
            position_cost=2.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.right_hand_task = mink.FrameTask(
            frame_name="right_gripper_link",
            frame_type="geom",
            position_cost=2.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        
        self.tasks = [
            self.pelvis_task,
            self.torso_task,
            self.posture_task,
            self.left_hand_task,
            self.right_hand_task,
        ]
        
        # ===== 碰撞避免 =====
        torso_geoms = ["torso_link1", "torso_link2", "torso_link3", "torso_link4"]
        collision_pairs = [
            (["left_gripper_link"], torso_geoms),
            (["right_gripper_link"], torso_geoms),
        ]
        
        collision_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.005,
            collision_detection_distance=0.15,
        )
        
        # 速度限制（防止 IK 输出过大速度导致仿真不稳定）
        velocity_limit = mink.VelocityLimit(self.model, {
            # 底盘速度限制
            "base_x": 0.1,    # 0.1 m/s
            "base_y": 0.1,    # 0.1 m/s
            "base_yaw": 0.5,  # 0.5 rad/s
            # 躯干关节速度限制
            "torso_joint1": 1.0,
            "torso_joint2": 1.0,
            "torso_joint3": 1.0,
            "torso_joint4": 1.0,
            # 左臂关节速度限制
            "left_arm_joint1": 2.0,
            "left_arm_joint2": 2.0,
            "left_arm_joint3": 2.0,
            "left_arm_joint4": 2.0,
            "left_arm_joint5": 2.0,
            "left_arm_joint6": 2.0,
            "left_arm_joint7": 2.0,
            # 右臂关节速度限制
            "right_arm_joint1": 2.0,
            "right_arm_joint2": 2.0,
            "right_arm_joint3": 2.0,
            "right_arm_joint4": 2.0,
            "right_arm_joint5": 2.0,
            "right_arm_joint6": 2.0,
            "right_arm_joint7": 2.0,
        })
        
        # ===== 约束 =====
        self.limits = [
            mink.ConfigurationLimit(self.model),
            collision_limit,
            velocity_limit,
        ]
        
        # 初始化状态
        self._initialized = False
    
    def initialize(self, initial_q: Optional[np.ndarray] = None):
        """
        初始化控制器状态
        
        Args:
            initial_q: 初始关节位置，如果为 None 则使用 "home" 关键帧
        """
        if initial_q is None:
            # 使用模型中的 home 关键帧
            self.configuration.update_from_keyframe("home")
        else:
            # 设置关节位置（18个：躯干+手臂）
            self.data.qpos[:self.N_BASE] = 0.0
            self.data.qpos[self.IDX_TORSO:self.IDX_TORSO + self.N_ARM_TOTAL] = initial_q[:self.N_ARM_TOTAL]
        
        # FK 更新
        mujoco.mj_forward(self.model, self.data)
        
        # 设置姿态任务目标
        self.posture_task.set_target_from_configuration(self.configuration)
        self.pelvis_task.set_target_from_configuration(self.configuration)
        self.torso_task.set_target_from_configuration(self.configuration)
        
        self._initialized = True
    
    def solve(self,
              left_target: Optional[np.ndarray],
              right_target: np.ndarray,
              dt: float = 0.01,
              damping: float = 1e-1,
              current_joint_positions: Optional[np.ndarray] = None) -> IKResult:
        """
        求解 IK
        
        Args:
            left_target: 左手目标 [x, y, z, qw, qx, qy, qz]，相对于 base_link（可选）
            right_target: 右手目标 [x, y, z, qw, qx, qy, qz]，相对于 base_link
            dt: 时间步长
            damping: QP 阻尼系数
            current_joint_positions: 当前关节位置（18个：躯干+双臂），如为None则使用内部状态
            
        Returns:
            IKResult: 包含关节位置和底盘速度
        """
        if not self._initialized:
            self.initialize(current_joint_positions)
        
        # 1. 更新内部状态（如果提供了外部关节位置）
        if current_joint_positions is not None:
            self.data.qpos[:self.N_BASE] = 0.0  # 底盘在原点
            self.data.qpos[self.IDX_TORSO:self.IDX_TORSO + self.N_ARM_TOTAL] = current_joint_positions[:self.N_ARM_TOTAL]
            mujoco.mj_forward(self.model, self.data)
        
        # 2. 设置手部目标（相对于 base_link = 世界原点）
        self.left_hand_task.set_target(self._array_to_se3(left_target))
        self.right_hand_task.set_target(self._array_to_se3(right_target))
        
        # 3. 求解 IK
        try:
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                self.solver,
                damping=damping,
                limits=self.limits
            )
            success = True
        except Exception as e:
            print(f"IK 求解失败: {e}")
            vel = np.zeros(self.nv)
            success = False
        
        # 调试：打印速度和任务误差
        self._debug_count = getattr(self, '_debug_count', 0) + 1
        if self._debug_count % 100 == 0:
            # 当前末端位置
            left_cur = self._get_geom_pose("left_gripper_link")[:3]
            right_cur = self._get_geom_pose("right_gripper_link")[:3]
            # 目标位置
            left_tgt = left_target[:3]
            right_tgt = right_target[:3]
            # 误差
            left_err = np.linalg.norm(left_tgt - left_cur)
            right_err = np.linalg.norm(right_tgt - right_cur)
            # 速度
            vel_max = np.abs(vel).max()
            vel_arm = vel[self.IDX_TORSO:self.IDX_TORSO + self.N_ARM_TOTAL]
            print(f"[IK调试] 左手误差={left_err:.4f}m 右手误差={right_err:.4f}m")
            print(f"         vel_max={vel_max:.6f} vel_arm_max={np.abs(vel_arm).max():.6f}")
            print(f"         左手: 目标={np.round(left_tgt,3)} 当前={np.round(left_cur,3)}")
            print(f"         右手: 目标={np.round(right_tgt,3)} 当前={np.round(right_cur,3)}")
        
        
        # 4. 积分更新内部状态
        self.configuration.integrate_inplace(vel, dt)
        
        # 5. 提取结果（只取躯干+双臂，不含轮子）
        base_velocity = vel[:self.N_BASE].copy()  # [vx, vy, wz]
        
        # qpos[9:27] = 躯干(4) + 左臂(7) + 右臂(7) = 18
        joint_positions = self.data.qpos[self.IDX_TORSO:self.IDX_TORSO + self.N_ARM_TOTAL].copy()
        
        return IKResult(
            joint_positions=joint_positions,
            base_velocity=base_velocity,
            success=success
        )
    
    def get_current_hand_poses(self, joint_positions: np.ndarray, debug: bool = False) -> tuple:
        """
        获取当前左右手位姿（相对于 base_link）
        
        Args:
            joint_positions: 关节位置（18个）
            debug: 是否打印调试信息
        
        Returns:
            (left_pose, right_pose): 每个是 [x, y, z, qw, qx, qy, qz]
        """
        self.data.qpos[:self.N_BASE] = 0.0
        self.data.qpos[self.IDX_TORSO:self.IDX_TORSO + self.N_ARM_TOTAL] = joint_positions[:self.N_ARM_TOTAL]
        mujoco.mj_forward(self.model, self.data)
        
        if debug:
            print("\n[FK调试] MuJoCo qpos 设置:")
            joint_names = [
                "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
                "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
                "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
            ]
            for i, name in enumerate(joint_names):
                mj_idx = self.IDX_TORSO + i
                print(f"  qpos[{mj_idx:2d}] {name:20s} = {self.data.qpos[mj_idx]:7.4f}")
        
        left_pose = self._get_geom_pose("left_gripper_link")
        right_pose = self._get_geom_pose("right_gripper_link")
        
        return left_pose, right_pose
    
    def _array_to_se3(self, arr: np.ndarray) -> mink.SE3:
        """将 [x, y, z, qw, qx, qy, qz] 转换为 SE3"""
        pos = arr[:3]
        quat = arr[3:7]  # [qw, qx, qy, qz]
        return mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(quat),
            translation=pos
        )
    
    def _get_geom_pose(self, geom_name: str) -> np.ndarray:
        """获取几何体位姿 [x, y, z, qw, qx, qy, qz]"""
        geom_id = self.model.geom(geom_name).id
        pos = self.data.geom_xpos[geom_id].copy()
        mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return np.concatenate([pos, quat])
