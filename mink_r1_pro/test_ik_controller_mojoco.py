"""
R1 Pro IK 控制器测试环境

使用 MuJoCo 可视化窗口测试 R1ProIKController 类。
拖动 mocap 目标球来控制机器人手臂，观察底盘速度输出。

操作说明：
- 拖动蓝色球：控制右手目标
- 按 ESC：退出
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

from r1pro_ik_wbc import R1ProIKController

# 模型路径
_HERE = Path(__file__).parent
_XML = _HERE / "r1_pro_ik_scene.xml"


def get_mocap_pose(data: mujoco.MjData, mocap_id: int) -> np.ndarray:
    """获取 mocap 体的位姿 [x, y, z, qw, qx, qy, qz]"""
    pos = data.mocap_pos[mocap_id].copy()
    quat = data.mocap_quat[mocap_id].copy()  # MuJoCo 格式: [qw, qx, qy, qz]
    return np.concatenate([pos, quat])


def main():
    # ===== 1. 创建控制器 =====
    print("加载模型...")
    controller = R1ProIKController(str(_XML))
    
    # 获取内部 model 和 data（用于可视化）
    model = controller.model
    data = controller.data
    
    # ===== 2. 获取 mocap 目标体的 ID =====
    # 只用右手目标（与原测试一致）
    right_hand_mocap_id = model.body("right_gripper_link_target").mocapid[0]
    
    # 检查是否有左手 mocap（可选）
    try:
        left_hand_mocap_id = model.body("left_gripper_link_target").mocapid[0]
        has_left_target = True
    except Exception:
        left_hand_mocap_id = None
        has_left_target = False
        print("注意: 模型中没有左手 mocap 目标，只控制右手")
    
    # ===== 3. 启动 MuJoCo 可视化器 =====
    print("启动可视化器...")
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # ===== 4. 初始化 =====
        controller.initialize()
        
        # 将 mocap 目标球移动到当前手部位置
        import mink
        mink.move_mocap_to_frame(model, data, "right_gripper_link_target", "right_gripper_link", "geom")
        if has_left_target:
            mink.move_mocap_to_frame(model, data, "left_gripper_link_target", "left_gripper_link", "geom")
        
        print("\n" + "=" * 50)
        print("IK 控制器测试环境已启动")
        print("=" * 50)
        print("操作: 拖动蓝色球控制右手目标")
        print("按 ESC 退出")
        print("=" * 50 + "\n")
        
        # ===== 5. 主控制循环 =====
        rate = RateLimiter(frequency=200.0, warn=False)
        frame_count = 0
        
        while viewer.is_running():
            # 从 mocap 读取目标位姿
            right_target = get_mocap_pose(data, right_hand_mocap_id)
            left_target = get_mocap_pose(data, left_hand_mocap_id) if has_left_target else None
            
            # 求解 IK（内部自动积分更新状态）
            result = controller.solve(
                left_target=left_target,
                right_target=right_target,
                dt=rate.dt
            )
            
            # solve() 内部已积分，直接更新前向运动学
            mujoco.mj_forward(model, data)
            mujoco.mj_camlight(model, data)
            
            # 定期打印底盘速度
            frame_count += 1
            if frame_count % 100 == 0:
                vx, vy, wz = result.base_velocity
                if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(wz) > 0.01:
                    print(f"底盘速度: vx={vx:.3f} m/s, vy={vy:.3f} m/s, wz={wz:.3f} rad/s")
            
            # 同步可视化
            viewer.sync()
            rate.sleep()
    
    print("\n测试结束")


if __name__ == "__main__":
    main()

