| 分类             | 项目名称                        | 描述                                                         |
| ---------------- | ------------------------------- | ------------------------------------------------------------ |
| **多模态大模型** | **PaLM-E**                      | Google研究的首个具身多模态模型，支持视觉+语言输入，直接输出机器人控制指令。 |
|                  | **RT-1 / RT-2**                 | Google DeepMind 提出的 Transformer-based 多模态 → 低级动作指令映射模型；RT-2 加入 Web 知识库以提升跨任务泛化能力。 |
|                  | **SayCan**                      | 将大语言模型用于高层规划，结合可调用的低级技能 API 让机器人完成复杂任务。 |
|                  | **VIMA**                        | MIT/FAIR 发布的桌面操作多模态策略模型，输入语言、视觉和历史动作，输出操作策略。 |
|                  | **RoboFlamingo**                | 上海 AI Lab 基于 Flamingo 的机器人专用多模态大模型，融合视觉、语言与控制数据。 |
|                  | **Octo**                        | Meta AI 的通用机器人策略大模型，支持多平台、多任务的跨域泛化。 |
| **行为与控制**   | **BEHAVIOR-1K**                 | Stanford 发布的大规模机器人行为库，包含 1000+ 日常交互任务，可用于仿真训练。 |
|                  | **ManiSkill**                   | 港中大团队开源的强化学习与仿真平台，专注复杂操作任务（抓取、装配、插拔等）的训练和评估。 |
|                  | **Open-X Embodiment**           | Google DeepMind 的跨平台数据与模型统一框架，兼容 RT 系列模型，支持多机器人数据集成。 |
|                  | **RoboAgent**                   | CMU 推出的少样本学习框架，能够在真实机器人上快速适应新任务。 |
| **系统与仿真**   | **ROS 2**                       | 最广泛使用的机器人中间件，提供硬件抽象、节点通信与任务调度机制。 |
|                  | **MoveIt 2**                    | 基于 ROS 2 的运动规划框架，支持机械臂和移动机器人的路径规划与碰撞检测。 |
|                  | **Isaac Sim / Isaac Gym**       | NVIDIA 提供的 GPU 加速仿真平台与强化学习环境，可与大模型对接进行高速训练。 |
|                  | **PyBullet / MuJoCo**           | 轻量级物理仿真引擎，适用于 Reinforcement Learning 和算法验证。 |
| **数据集**       | **RT-X Dataset**                | Google 组织的大规模多机器人演示数据集，涵盖多种平台与任务。  |
|                  | **Open X-Embodiment Dataset**   | 跨 22+ 机器人平台的 1.4M+ 示范数据，用于多任务和多机器人的统一训练。 |
|                  | **EPIC-KITCHENS**               | 大规模真实厨房操作视频数据集，适合具身智能和视觉理解研究。   |
|                  | **BridgeData v2**               | Berkeley 发布的真实机器人示教数据集，包含多种抓取与操作示例。 |
| **国内项目**     | **RoboFlamingo（上海 AI Lab）** | 多模态机器人大模型研究，融合视觉、语言和控制；兼顾学术与工程实践。 |
|                  | **XSkill（清华）**              | 清华大学结合语言模型与低级动作库，研究机器人高层任务规划与自然交互。 |
|                  | **RoboMind（北京智源）**        | 智源研究院的多模态具身智能项目，面向实际场景的机器人感知与控制。 |
|                  | **昇腾机器人（华为）**          | 华为基于 MindSpore 与昇腾硬件平台的机器人操作系统及开发套件。 |

## **五、典型demo**

| 仓库名称                       | 功能简介                                    | 上手步骤                                                     |
| ------------------------------ | ------------------------------------------- | ------------------------------------------------------------ |
| **robotics_transformer**(RT-1) | Google RT-1 多模态→动作指令，支持仿真到真机 | 1. Clone2. 安装依赖：`pip install -r requirements.txt`3. 下载示例模型4. 在 Gazebo/Isaac Sim 中运行 demo |
| **say-can**                    | SayCan 规划+技能执行框架                    | 1. Clone2. 准备技能 API3. `pip install -e .`4. 运行交互脚本  |
| **vima**                       | MIT VIMA 桌面操作多模态策略                 | 1. Clone2. 下载预训练权重3. 安装依赖并启动 PyBullet 仿真4. 运行 `demo.ipynb` |
| **roboflamingo**               | 上海 AI Lab RoboFlamingo                    | 1. Clone2. 安装环境：`conda env create -f env.yml`3. 下载视觉+控制模型4. 运行 `python demo.py` |
| **open-x-embodiment**          | 多平台数据+模型一体化                       | 1. Clone2. 安装：`pip install open_x_embodiment`3. 在 ROS2/Gazebo 中加载示例 |
| **unitree-ros**                | Unitree G1/Comp ROS 驱动与示例              | 1. Clone 到 ROS2 工作区2. `colcon build`3. 启动 `ros2 launch unitree_g1 bringup.launch.py`4. 运行点控／行走 demo |

mujoco世界仿真

[openpi](https://github.com/Physical-Intelligence/openpi)

**[umi](https://github.com/umijs/umi)**

[字节 GR-3 4B](https://seed.bytedance.com/zh/blog/seed-research-%E9%80%9A%E7%94%A8%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%A8%A1%E5%9E%8B-gr-3-%E5%8F%91%E5%B8%83-%E6%94%AF%E6%8C%81%E9%AB%98%E6%B3%9B%E5%8C%96-%E9%95%BF%E7%A8%8B%E4%BB%BB%E5%8A%A1-%E6%9F%94%E6%80%A7%E7%89%A9%E4%BD%93%E5%8F%8C%E8%87%82%E6%93%8D%E4%BD%9C)

[WorldVLA](https://github.com/alibaba-damo-academy/WorldVLA)

[TriVLA](https://arxiv.org/abs/2507.01424)
