# RK3566 差速移动机器人（robot-full）

基于 **RK3566（香橙派）+ STM32F103** 的 ROS1 差速移动机器人工程：涵盖相机感知、YOLOv6 目标检测、激光雷达、视觉–激光跟随、底盘串口控制、导航，以及端侧语音 Agent 交互。仓库对应 GitHub：`dantemike/rk3566-robot`。

---

## 硬件组成

| 模块 | 型号 / 接口 | 说明 |
|---|---|---|
| 主控（上位机） | **RK3566 香橙派** | 跑 Linux + ROS1 Noetic、检测、导航、Agent |
| 底盘 MCU（下位机） | **STM32F103** | 电机 PWM、编码器采集、串口协议、心跳停车 |
| 电机驱动 | **TB6612** | TIM1 PWM + GPIO 方向脚驱动左右轮 |
| 编码器 | TIM2 / TIM4 正交编码 | 左右轮速度 / 里程反馈 |
| 串口链路 | 香橙派 **`/dev/ttyS2`** ↔ MCU **USART3**，**115200 8N1** | 下发 PWM，回传编码器 |
| USB 相机 | **`/dev/video0`**，默认 **1280×720** | MJPEG 采集，frame `camera_link` |
| 激光雷达 | **YDLIDAR**（三角测距 / 单通道），**`/dev/ttyS9`**，115200 | 发布 `/scan`，frame `laser_link` |
| NPU | RK3566 内置 NPU | 通过 **RKNN Runtime** 跑 `.rknn` 检测模型 |
| 语音输入 | USB 麦克风（UAC） | 内核 USB Audio + ALSA，应用层 PyAudio 采 PCM |
| 交互外设 | LCD 表情、云台/舵机（PWM/SPI 等） | Agent FunctionCall 控制 |
| 底盘形态 | 差速两轮 | Linux 侧速度环 PID + 逆解，MCU 侧执行 |

---

## 功能总览

```
USB 相机 → 硬解 → YOLOv6(NPU) → 检测框
YDLIDAR  → /scan ────────────────┘
                ↓
         object_track（几何后融合）→ /cmd_vel
                ↓
         base_control（PID + 串口）→ STM32 → 电机
                ↓
              /odom、导航（move_base 等）

Agent：麦克风 → ASR → LLM → FunctionCall → 导航/跟随/云台/LCD/TTS
```

---

## 软件功能（按功能包）

### 1. 感知与检测

| 包 | 功能 |
|---|---|
| `usb_camera` | V4L2 采集 USB 相机，发布压缩图 |
| `img_decode` | Rockchip **MPP** 硬解 MJPEG → RGB，供推理 |
| `img_encode` | 图像编码（检测可视化等链路） |
| `rknn_yolov6` | **YOLOv6** 转 `.rknn` 后经 **RKNN Runtime** 在 NPU 推理；发布 `/camera/image_det`、`/ai_msg_det`。x86 编译时可用 OpenCV **Haar** 做人脸软检测对比，真车 aarch64 走 YOLO |
| `ydlidar` | YDLIDAR SDK + ROS 节点，发布 `sensor_msgs/LaserScan` |

### 2. 目标跟随

| 包 | 功能 |
|---|---|
| `object_track` | **视觉语义 + 激光测距几何后融合**：同步检测图与框，选最大 `person`；激光经 TF 投到图像，框内取最近距离；比例控制发 `/cmd_vel`；`/enable_tracking`、`/enable_lidar` 可开关 |

### 3. 底盘与 MCU

| 包 / 目录 | 功能 |
|---|---|
| `base_control` | 订 `/cmd_vel`，差速逆解 + 速度环 PID，串口发 `CmdData`（PWM），收 `McuData`（编码器），发 `/odom` |
| `robot/MCU/1-LED` | STM32 工程：TIM1 PWM、编码器、USART3、SysTick≈20 ms 周期回传、约 100 ms 无命令心跳停车 |

协议要点：帧头 `DA`，帧尾 `TA\r\n`，约 11 字节结构体，`pack(1)` 对齐。

### 4. 导航与模型

| 包 | 功能 |
|---|---|
| `robot_navigation` | 导航相关 launch / maps / 配置（move_base、代价地图等） |
| `test11_description` | 机器人 URDF / 模型描述 |
| `pkg_launch` | 一键启动：`all.launch`（相机–解码–检测–编码–雷达–跟随；底盘 launch 默认注释，可按需打开）、`urdf.launch` |

### 5. 语音 Agent

| 目录 | 功能 |
|---|---|
| `robot/src/agent` | **在线**：流式 ASR、唤醒词、云端 LLM、TTS、FunctionCall（导航/跟随/云台/LCD 等） |
| | **离线向**：小模型微调/量化、NPU 部署、双缓冲与时延优化（完整离线栈亦可见完整仓 `offline-agent`） |
| | 采音：USB 麦 → ALSA → PyAudio（16 kHz PCM） |

### 6. 通信优化（可选）

| 目录 | 功能 |
|---|---|
| `shm-transport` / `shm_transport` | 同机大数据经 **共享内存** 传载荷，ROS 传 handle，降低大图/雷达拷贝开销。完整包需含 `package.xml`、`CMakeLists.txt` 等；仅头文件时不可单独 catkin 编译 |

### 7. 其他

| 项 | 说明 |
|---|---|
| `ai_msgs` | 检测结果等自定义消息 |
| `config` | 公共配置 |
| `web_controller.py` / `templates` | Web 相关控制页面（若使用） |

---

## 目录结构（简要）

```text
robot-full-20261110/
├── README.md
├── robot/                    # ROS 工作空间根
│   ├── src/                  # 功能包
│   │   ├── usb_camera/
│   │   ├── img_decode/ img_encode/
│   │   ├── rknn_yolov6/
│   │   ├── ydlidar/
│   │   ├── object_track/
│   │   ├── base_control/
│   │   ├── robot_navigation/
│   │   ├── agent/
│   │   ├── shm-transport/    # 共享内存传输（结构以完整包为准）
│   │   └── pkg_launch/
│   └── MCU/1-LED/            # STM32 底盘固件工程
└── ...
```

---

## 环境与依赖（概要）

- **系统**：Linux aarch64（板端），ROS1 **Noetic** 推荐  
- **编译**：`catkin_make`；板端检测需链接 `rknnrt` / `rga` 等  
- **模型**：YOLOv6 → ONNX → **RKNN-Toolkit** 转 `.rknn`（转换在 PC，推理在板）  
- **Python Agent**：见 `agent/requirements.txt`；云端 API Key 等勿提交仓库  

---

## 快速启动（板端示意）

```bash
cd robot
source devel/setup.bash   # 或对应安装空间

# 感知 + 检测 + 雷达 + 跟随（见 all.launch）
roslaunch pkg_launch all.launch

# 底盘（按需取消 all.launch 中注释，或单独启动）
roslaunch base_control base_control.launch

# 语音 Agent（示例）
cd src/agent && ./start_agent.sh
```

串口权限、雷达/相机设备节点名请按实车修改 launch 参数。

---

## 典型话题

| 话题 | 含义 |
|---|---|
| `/image_raw/compressed` | 相机压缩图 |
| `/camera/image_det`、`/ai_msg_det` | 检测可视化与检测框 |
| `/scan` | 激光 |
| `/cmd_vel` | 速度指令 |
| `/odom` | 里程计 |
| `/enable_tracking` | 跟随开关 |

---

## 说明与边界

- MCU 工程含 ST 标准外设库，业务逻辑在 `main` / `usart3` / `pwm` / `encoder`；部分为继承工程上的联调与协议对接。  
- 真车检测为 **YOLO + RKNN**；Haar 仅为非 ARM 环境软检测路径。  
- `shm_transport` 以完整 ROS 包形式使用；散落头文件仅便于阅读。  
- 大模型训练/量化在 PC 完成，板上做 Runtime 推理与业务编排。  

---

## License / 用途

教学与实习展示用移动机器人全栈参考工程。使用第三方 SDK（YDLidar、RKNN、OpenCV 等）时请遵守各自许可协议。
