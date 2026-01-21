# Isaac Lab R1 Pro IK & Swerve Control

本项目实现了 **R1 Pro 人形机器人** 的全身运动控制，核心是将 Isaac Lab 中的机器人状态与 ROS 2 Swerve 底盘控制器打通，并结合全身逆运动学 (WBC) 实现目标追踪。

## 快速运行

### 1. 启动 ROS 2 底盘控制器
```bash
ros2 launch swerve_controller r1pro_swerve.launch.py
```
### 2. 启动仿真
(在另一个终端中运行)
```bash
cd isaaclab_r1_pro_ik
python main.py --num_envs 1
```
*   **操作**: 按住 `Shift` + `左键` 拖动场景中的**黄色球体**，机器人会全身协调运动去追踪它。

## ROS 2 接口说明

| 话题 | 方向 | 作用 |
| :--- | :--- | :--- |
| `/cmd_vel` | Pub (Sim -> ROS) | IK 算出的底盘期望速度。 |
| `/Swerve_wheel_jointcommand` | Sub (ROS -> Sim) | 接收解算后的车轮控制指令（转向角+轮速）。 |
| `/Swerve_wheel_jointstates` | Pub (Sim -> ROS) | 发送车轮的真实状态反馈给控制器。 |
