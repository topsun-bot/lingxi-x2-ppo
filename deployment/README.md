# Deployment · 策略与机器人之间的明确边界

[训练](../training/) · [模型](../models/history2_hybrid_v1/) · [English](#english-notes)

这里包含可独立测试的 C++ 策略核心，以及面向 ROS 2 / AIMDK HAL 的轻量适配器。训练框架无需安装到机器人：部署端仅依赖 CPU ONNX Runtime、ROS 2 与机器人上已有的官方消息接口。

> **发布状态**：策略核心沿用本项目的双帧观测和动作转换实现；ROS 2 节点是为开源独立整理的新实现，不包含原厂未标注许可的控制器源码。历史仿真和真机结果不代表这个新节点已完成真机回归。默认入口只读传感器、运行推理，不创建电机指令发布器。

## 结构

```text
deployment/
├── include/                         # 无 ROS 依赖的策略契约与观测构造
├── src/x2_velocity_policy_core.cc    # 104 / 208 维观测、动作到关节目标
├── src/ros2_controller.cpp           # 原创 ROS 2 / ORT / HAL 适配
├── config/                          # dry-run、仿真、显式硬件示例
├── scripts/                         # 构建、默认 dry-run、模型检查与实体手柄
└── tests/                           # 可在普通 Linux CI 运行的核心测试
```

本目录不打包 AIMDK SDK、`aimdk_msgs`、厂商仿真器、共享库、SSH 密码或某一台机器的部署路径。官方接口请通过 [AimDK X2 文档](https://x2-aimdk.agibot.com/zh-cn/latest/quick_start/index.html) 单独取得。

## 先运行无需机器人的检查

需要 CMake ≥ 3.16 和支持 C++17 的编译器：

```bash
bash deployment/scripts/test_core.sh
```

检查项包含重力投影、单帧 / 双帧观测顺序、首帧复制和历史重置、命令边界、目标角与速率边界、非有限数拒绝、v1 / v2 metadata 解析。Release 构建也保留测试断言。

ONNX CPU 冒烟检查：

```bash
python -m pip install numpy onnxruntime
python deployment/scripts/inspect_model.py
```

默认模型是仓库根目录的 `models/history2_hybrid_v1/policy.onnx`，部署目录不保存第二份二进制。冒烟检查只证明模型可以加载和推理，不证明步态稳定。

## 训练 → 部署契约

| 项目 | 契约 |
| --- | --- |
| 输入 / 输出 | `obs: float32[1,208]` → `actions: float32[1,31]` |
| 兼容单帧 | `agibot_x2_velocity_v1` 使用 104 维；v2 固定两帧 |
| 历史布局 | **逐项**上一帧、当前帧；不是整个 104 维向量简单拼接 |
| 观测项 | 角速度 3、投影重力 3、命令 3、相位 2、关节位置 31、速度 31、上次动作 31 |
| 坐标约定 | **pelvis / root** 坐标角速度及姿态推导的重力，ROS 四元数转换为 `wxyz`；关节位置减默认角 |
| 静止命令 | 命令范数 `< 0.1` 时相位置零；hybrid 路由已封装在 ONNX 内 |
| 频率 | metadata `policy_dt=0.02`，目标推理 50 Hz；关节发送目标 500 Hz |
| 动作映射 | `q_des = q_default + action_scale × action`，随后执行明确配置的目标边界 |
| PD 增益 | 模型 metadata 的 `joint_stiffness` / `joint_damping` × `gain_scale` |

必须随 ONNX 保留的 metadata：

```text
metadata_contract, observation_names, observation_dims,
observation_history_length (v2), joint_names, default_joint_pos,
action_scale, joint_stiffness, joint_damping,
joint_limit_lower, joint_limit_upper, policy_dt, phase_period
```

节点拒绝名称、dtype、shape、历史长度或 metadata 不匹配的模型。关节按名称关联 HAL 反馈，再按模型 metadata 顺序生成观测；禁止依赖消息中的偶然顺序。

`action_scale` 已由训练导出。不要在部署端再做一遍归一化，也不要把动作强制截为 `[-1,1]`。当前配置用 `action_clip: 100`、`action_scale_factor: 1`、`max_target_velocity: 1000`，用于保留训练动作；机械范围仍来自模型 metadata。该节点不会额外引用一个来源不明的机器人参数表，实际型号 / 固件的物理限制须与训练资产核对。

### 核实 IMU 的实际来源与坐标约定

训练中的 `imu_0` 绑定 pelvis/root。官方默认输入是 `/aima/hal/imu/torso/state`；**topic 名称不能推定传感器安装在哪个刚体**。官方仿真配置将该通道连接到 body orientation/angular velocity，另有独立 chest 通道，因此不能仅凭 `torso` 字样认定它与训练不一致。实际机器人仍应核实传感器来源、固件映射、轴向和标定。

`dry_run.yaml` 使用官方 topic，并以 `imu_frame: unverified` 表示本次发布没有重新完成该机器的传感器来源验收。所有开启输出的配置要求显式设置 `imu_frame: pelvis`，声明姿态和角速度符合训练的 pelvis/root 约定。硬件示例保留 `unverified`，**在完成该确认前会拒绝输出**。这个字段是集成者对数据语义的确认，不会旋转数据，也不要求原厂 `header.frame_id` 使用某个固定字符串。

如果核实实际来源确为另一活动刚体，才需要额外的运动学 / 坐标转换；有腰部相对运动时还要处理角速度补偿，不能仅靠重命名 topic 或静态旋转代替。适配器没有实现这个条件性的转换，也不把未经证实的坐标错配当作历史实机失稳的根因。

## 构建 ROS 2 适配器

面向 Ubuntu 22.04 / ROS 2 Humble。需要 `rclcpp`、`sensor_msgs`、`std_srvs`、`yaml-cpp`、`colcon`、外部官方 `aimdk_msgs` 和同 CPU 架构的 ONNX Runtime C/C++ 包。CPU 推理不需要 CUDA，也不需要 PyTorch。

```bash
# 示例占位路径：替换为实际安装位置。不要把 SDK / ORT 二进制提交进仓库。
export AIMDK_MSGS_PREFIX=/opt/aimdk/aimdk_msgs
export ONNXRUNTIME_ROOT=/opt/onnxruntime
export ROS_SETUP=/opt/ros/humble/setup.bash
bash deployment/scripts/build_ros2.sh
```

`AIMDK_MSGS_PREFIX` 下应存在 `share/aimdk_msgs/cmake/aimdk_msgsConfig.cmake`。PC2 使用 aarch64 Runtime；普通开发机使用 x86_64 Runtime。构建输出默认放在 `deployment/workspace/`，可通过 `X2_ROS_WORKSPACE` 覆盖。

外部 SDK 和 ROS 环境已正确 source 后，可运行隔离的合成消息回归：

```bash
python deployment/tests/ros2_smoke.py \
  --controller deployment/workspace/install/lingxi_x2_deployment/lib/lingxi_x2_deployment/x2_policy_controller
```

该测试强制 localhost、独立 ROS domain，仅发布合成 HAL 反馈；检查 dry-run 无电机 publisher、只读急停服务、策略输出、软件急停锁存、按键不能误解除急停，以及启动时已按住的按钮不会误使能。它不执行物理仿真，也不连接机器人。

## 默认入口：真实反馈 dry-run

```bash
export AIMDK_MSGS_PREFIX=/opt/aimdk/aimdk_msgs
export ONNXRUNTIME_ROOT=/opt/onnxruntime
export ROS_DOMAIN_ID=0  # 必须与实际系统一致；这里是示例
# 有线 DDS 设置由现场提供，例如 FASTRTPS_DEFAULT_PROFILES_FILE。
bash deployment/scripts/run.sh
```

预期先出现 `DRY RUN: no motor command publishers exist`；收到全部 31 关节及躯干 IMU 后才执行推理。节点用单调时钟判断数据年龄，日志每两秒输出一次。等待状态、姿态不符或 metadata 错误均会明确显示，不能用“进程还活着”代替就绪判断。

`MODEL_PATH` 可指向其它符合契约的 ONNX；`X2_CONFIG` 选择其它配置。ROS 名称可通过常规 remap 覆盖，例如 `... run.sh --ros-args -r /joy:=/operator/joy`。

## 数据流与现场接口

| 方向 | ROS 2 topic / service | 类型 |
| --- | --- | --- |
| 输入 | `/aima/hal/joint/{leg,waist,arm,head}/state` | `aimdk_msgs/msg/JointStateArray` |
| 输入 | `imu_topic`，默认 `/aima/hal/imu/torso/state`；数据语义须匹配 pelvis/root | `sensor_msgs/msg/Imu` |
| 输入 | `/joy` | `sensor_msgs/msg/Joy` |
| 输出，默认不存在 | `/aima/hal/joint/{leg,waist,arm,head}/command` | `aimdk_msgs/msg/JointCommandArray` |
| 软件急停 | `/x2/policy/estop` | `std_srvs/srv/Trigger` |

关节、IMU、手柄有独立回调组，三线程 executor 调度。控制与推理在同一个互斥定时回调中执行，ONNX Runtime 单线程计算；这不是硬实时调度器。所有 31 关节和所配置的 IMU 都要求持续更新，不默认忽略上肢反馈。未收到首份完整状态时不发位置目标；运行后反馈超时、关节错误、非有限值或超倾角会锁存故障并发布阻尼。`disarm` 可清除普通故障后重新选择模式；软件急停只能通过明确重启解除。

策略按单调时钟绝对 deadline 累计 20 ms 周期，避免 2 ms 发送定时器抖动被累积进每次推理周期。错过周期会计入 `skipped` 并跳过，不会在同一时刻补算多个历史帧；健康日志同时给出实际推理间隔峰值。正常目标仍是 50 Hz，需用实际运行记录确认其达到程度。

本节点不自动切换 `Develop_MC`，不停止原厂进程，也不恢复 `Business`。这些系统状态必须按当前官方固件流程在现场管理，避免多个控制器同时写 HAL。节点退出和通信中断后的驱动行为由实际机器人 / 固件决定，不能把 Linux 进程退出当作电机断电。

## 实体手柄与软件急停

可使用 ROS 2 `joy` 节点，或运行本仓库的 Linux joydev 桥：

```bash
source /opt/ros/humble/setup.bash
python deployment/scripts/joy_bridge.py --device /dev/input/js0
ros2 topic echo /joy
```

桥直接转发物理按钮；不伪造持续 deadman，不自动站立，不发送速度序列。先确认具体手柄的轴号 / 按钮号，再修改 YAML。示例映射如下（Linux 枚举可能不同）：

| 操作 | 示例索引 | 行为 |
| --- | --- | --- |
| 持续使能 | button 4 | 硬件输出模式下必须保持 |
| disarm | button 0 | 不发布电机指令，清除普通故障 |
| 软件急停 | button 1 | 锁存阻尼，明确重启后才可再使能 |
| 准备默认姿态 | button 2 | 从当前关节位置平滑过渡到 metadata 默认角 |
| RL | button 3 | 进入策略，硬件入口检查默认姿态误差 |
| 前后 / 横移 / 转向 | axes 1 / 0 / 3 | 按 YAML 符号和最大命令速度映射 |

首次收到的按钮状态只用于初始化，不将开机时已经按住的按钮当作启动边沿。按键释放后重新按下才触发状态切换。松开 deadman 或手柄数据超时会进入阻尼故障。

```bash
ros2 service call /x2/policy/estop std_srvs/srv/Trigger '{}'
```

软件急停使 `Kp=0`、`Kd=damping_gain` 并持续输出阻尼；它不会切断驱动电源，也不清理系统级进程。实体急停仍由原厂硬件链路提供。

## 显式硬件与仿真配置

| 配置 | 行为 |
| --- | --- |
| `dry_run.yaml` | 默认，只读观测 / 推理，不创建电机 publisher |
| `simulation.yaml` | 发布仿真 HAL 命令，启动即 RL；需独立仿真 ROS domain |
| `hardware.example.yaml` | 显式硬件示例，从 disarmed 启动，需实体 deadman / 模式按钮 |

硬件示例需先核实上文 IMU 来源 / 约定并将 `imu_frame` 改为 `pelvis`，再同时设置配置并传递 `--hardware`。保留 `unverified` 时以下命令会明确拒绝输出；它也不会替代现场的 SDK 状态切换和型号核对：

```bash
export X2_CONFIG="$PWD/deployment/config/hardware.example.yaml"
bash deployment/scripts/run.sh --hardware
```

硬件示例中的低速范围只缩小输入命令，不改变已训练的关节动作缩放。`gain_scale=1.0` 对齐模型 metadata；不要把历史某次 `1.05` 实验当成通用改进。真机上新节点的时序、传感器坐标、手柄映射和停机链路仍需逐项回归。

对 ROS HAL 仿真器使用 `simulation.yaml`；本仓库主要仿真验证与录像入口在 [training](../training/)，不依赖厂商闭源仿真器。

## English notes

The standalone C++ core implements the exported policy contract: metadata-driven joint ordering, term-major two-frame observations, gravity projection, and action-to-position conversion. `scripts/test_core.sh` needs neither ROS nor a robot.

The optional ROS 2 adapter is a new project-owned implementation. Obtain the official AIMDK message package separately and set `AIMDK_MSGS_PREFIX` and `ONNXRUNTIME_ROOT`. Run `scripts/build_ros2.sh`, then `scripts/run.sh`; the default profile never creates motor-command publishers. Model binaries live once, at `../models/history2_hybrid_v1/policy.onnx`.

This new ROS node has not completed a hardware regression campaign. Historical policy / deployment results must not be treated as validation of the rewritten node. It does not migrate robot system state or launch a motion sequence. Hardware use requires an explicit profile, `--hardware`, a disarmed start, and a live physical deadman. The E-stop service and damping button latch software damping; they do not cut motor power.

The policy requires **pelvis/root-frame** orientation and angular velocity. The official default input remains `/aima/hal/imu/torso/state`: a topic name does not identify the physical sensor body, and no frame mismatch is asserted here. Confirm its actual source, axes and calibration before setting `imu_frame: pelvis`; the default hardware example marks it unverified and blocks output until that confirmation. A moving-body transform would only be necessary if verification identifies a different source body; such a transform is not included.
