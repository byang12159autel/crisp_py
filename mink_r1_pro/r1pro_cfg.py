"""
R1_PRO Environment Configuration
--------------------------------
定义 R1 Pro 全向轮人形机器人的场景、动作和观测配置
"""
# =====================================================================
# 导入模块 (统一使用 isaaclab.*)
# =====================================================================
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


# =====================================================================
# 1. 场景配置 (加载 R1_PRO USD)
# =====================================================================
@configclass
class R1ProSceneCfg(InteractiveSceneCfg):
    # 必需字段：环境数量和间距
    num_envs: int = 1           # 并行环境数量
    env_spacing: float = 4.0    # 环境间距（米）
    
    # 机器人资产
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # [TODO] 请修改为您的 R1_PRO 机器人的真实 USD 路径
            usd_path="/home/ubuntu/Desktop/Collected_r1pro_swerve_controller/r1pro_v3.usd",
        ),
        # 初始状态配置
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # 机器人初始位置 (x, y, z)
            # 所有关节归零（正则表达式匹配）
            joint_pos={
                "torso_joint.*": 0.0,       # 躯干关节 (4个)
                "left_arm_joint.*": 0.0,    # 左臂关节 (7个)
                "right_arm_joint.*": 0.0,   # 右臂关节 (7个)
                "steer_motor_joint.*": 0.0, # 转向电机 (3个)
                "wheel_motor_joint.*": 0.0, # 驱动轮归零
            },
        ),
        # 执行器配置（参考 r1_pro.xml 中的 kp/kv 值）
        actuators={
            # 躯干关节 (高刚度)
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint.*"],
                stiffness=5000.0,  # kp=5000
                damping=500.0,     # kv=500-800
            ),
            # 手臂关节1-2 (提高增益，加快响应)
            "arm_joint_1_2": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint1", "left_arm_joint2", "right_arm_joint1", "right_arm_joint2"],
                stiffness=800.0,   # 提高 kp: 300→800
                damping=80.0,      # 提高 kv: 30→80
            ),
            # 手臂关节3-4 (提高增益)
            "arm_joint_3_4": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint3", "left_arm_joint4", "right_arm_joint3", "right_arm_joint4"],
                stiffness=400.0,   # 提高 kp: 175→400
                damping=40.0,      # 提高 kv: 7.5→40
            ),
            # 手臂关节5-7 (提高增益)
            "arm_joint_5_7": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
                                  "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"],
                stiffness=100.0,   # 提高 kp: 25→100
                damping=10.0,      # 提高 kv: 0.5→10
            ),
            # 转向电机
            "steer_motors": ImplicitActuatorCfg(
                joint_names_expr=["steer_motor_joint.*"],
                stiffness=200.0,    # kp=10
                damping=1.0,       # kv=1
            ),
            # 驱动轮 (速度控制)
            # 注意：轮子质量 50kg，需要较高的 damping 才能驱动
            "wheel_motors": ImplicitActuatorCfg(
                joint_names_expr=["wheel_motor_joint.*"],
                stiffness=0.0,     # 速度控制无刚度
                damping=10.0,    # kv=100, 驱动 50kg 轮子需要较大增益
            ),
        },
    )
    # 地面（使用内置地面平面，无需外部 USD）
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.5, 0.8, 1.0)),  # 浅蓝色地面
    )
    # 光照
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
    )
    # 柠檬目标球体（使用 AssetBaseCfg 定义，支持 GUI 拖动）
    # 注意：AssetBaseCfg 只负责生成资产，不会被 RigidObjectManager 接管，
    # 因此可以在仿真中通过 GUI 拖动，代码通过 USD Prim 读取其位置。
    lemon = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/lemon",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.0)),  # 黄色
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.20961, 0.5687, 0.85),
            rot=(0.5, -0.5, 0.5, 0.5),  # [w, x, y, z]
        ),
    )


# =====================================================================
# 2. 动作空间配置 (Action Split)
# =====================================================================
@configclass
class R1ProActionsCfg:
    # --- A. 上身 (Mink IK) -> 位置控制 ---
    # 躯干 + 双臂（共 4 + 7 + 7 = 18 个关节）
    # 注意：必须明确指定顺序，与 IK 输出一致（躯干 + 左臂全部 + 右臂全部）
    upper_body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
            "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
            "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
            "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
        ],
        use_default_offset=False,  # 外部 IK Solver 输出绝对角度
    )
    # --- B. 底盘转向 (Swerve Steer) -> 位置控制 ---
    # 3个转向电机
    chassis_steer = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["steer_motor_joint.*"],
        use_default_offset=False,
    )
    # --- C. 底盘驱动 (Swerve Drive) -> 速度控制 ---
    # 3个驱动轮，无限旋转用速度控制
    chassis_drive = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_motor_joint.*"],
        scale=1.0,
    )
    # --- D. 夹爪 -> 位置控制 ---
    # 左右各2个手指关节，共4个
    gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_gripper_finger_joint1", "left_gripper_finger_joint2",
            "right_gripper_finger_joint1", "right_gripper_finger_joint2",
        ],
        use_default_offset=False,
    )


# =====================================================================
# 3. 观测配置 - 定义智能体能观察到的状态信息
# =====================================================================
@configclass
class R1ProObservationsCfg:
    """观测配置：包含关节位置和速度信息"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络使用的观测组"""
        # 所有关节的相对位置（相对于默认位置）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 所有关节的相对速度
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False  # 不启用观测噪声
            self.concatenate_terms = True   # 将所有观测项拼接成单一向量

    # 观测组实例化
    policy: PolicyCfg = PolicyCfg()


# =====================================================================
# 4. 总环境配置
# =====================================================================
@configclass
class R1ProEnvCfg(ManagerBasedEnvCfg):
    """R1 Pro 机器人环境配置"""
    # 场景配置
    scene: R1ProSceneCfg = R1ProSceneCfg()
    # 动作配置
    actions: R1ProActionsCfg = R1ProActionsCfg()
    # 观测配置
    observations: R1ProObservationsCfg = R1ProObservationsCfg()

    def __post_init__(self):
        # 频率对齐设置 (目标 100Hz)
        self.decimation = 10  # 策略执行频率 = 物理频率 / 2
        self.sim.dt = 0.005  # 物理步长 200Hz
        self.sim.render_interval = self.decimation
