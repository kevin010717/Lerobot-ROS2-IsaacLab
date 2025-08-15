| 策略类别                  | 代表方法                                                     | 核心特点                                                     |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 非Transformer语言指令控制 | CLIPort / BC-Z / MCIL / HULC(++) / UniPi                     | 双流架构；零样本泛化；分层计划；视听对齐；视频生成→逆动力学解动作 |
| 基于Transformer的控制策略 | Interactive Language / Hiveformer / Gato / RoboCat / RT-1 / Q-Transformer / RT-Trajectory / ACT / RoboAgent / RoboFlamingo | 统一标记；自我改进；EfficientNet 编码；自回归 Q；轨迹草图条件；动作分块；预训练 VLM |
| 多模态指令的控制策略      | VIMA / MOO                                                   | 多模态提示；图像＋语言联合；新对象指定（手指指向／GUI 点击） |
| 具有 3D 视觉的控制策略    | RoboUniView / VER / PerAct / Act3D / RVT, RVT-2              | 点云／体素输入；多视角重构；自适应分辨率；6-DoF 增强         |
| 基于扩散的控制策略        | Diffusion Policy / SUDD / Octo / MDT / RDT-1B                | 扩散模型（DDPM／DiT）；时序扩散；LLM 生成→数据蒸馏；模块化设计 |
| 3D 视觉＋扩散策略         | DP3 / 3D Diffuser Actor                                      | 3D 体素 + 扩散生成                                           |
| 运动规划的控制策略        | Language costs / VoxPoser / RoboTAP                          | 自然语言指令→成本图；LLM + VLM 生成代码；阶段化视觉伺服      |
| 基于点的动作控制策略      | PIVOT / RoboPoint / ReKep                                    | VLM 关键点选择；可用性预测；3D 投影→约束优化                 |
| 大规模 VLA                | RT-2 / RT-H / RT-X / OpenVLA / π0                            | 互联网规模微调；动作层次；LoRA／量化；混合专家架构           |

| 数据集        | 机械臂平台   | 数据类型   | 指令形式        | 任务类型 | 对象/指令数 | 环境数 | 样本数    |
| ------------- | ------------ | ---------- | --------------- | -------- | ----------- | ------ | --------- |
| MIME          | Baxter       | RGB + 深度 | 演示 (Demo)     | 12 种    | 20 种对象   | 1      | 8.3 K     |
| RoboTurk      | Sawyer       | RGB        | —               | 2 种     | —           | 1      | 2.1 K     |
| RoboNet       | 7 种平台     | RGB        | 目标图像        | —        | —           | 10     | 162 K     |
| MT-Opt        | 7 种机器人   | RGB        | 语言指令        | 2 种     | 12 种对象   | 1      | 800 K     |
| BC-Z          | Everyday     | RGB        | 语言指令 + 演示 | 3 种     | 100 种对象  | 1      | 25.9 K    |
| RT-1-Kitchen  | Everyday     | RGB        | 语言指令        | 12 种    | 700 + 对象  | 2      | 130 K     |
| MOO           | Everyday     | RGB        | 多模态指令      | 5 种     | 106 条指令  | 1      | 59.1 K    |
| VIMA          | UR5          | RGB        | 多模态指令      | 17 种    | 29 条指令   | 1      | 650 K     |
| RoboSet       | Franka Panda | RGB + 深度 | 语言指令        | 12 种    | 38 种对象   | 11     | 98.5 K    |
| BridgeData V2 | WidowX 250   | RGB + 深度 | 语言指令        | 13 种    | 100 + 指令  | 24     | 60.1 K    |
| RH20T         | 4 种平台     | RGB + 深度 | 语言指令        | 42 种    | 147 种对象  | 7      | 110 K +   |
| DROID         | Franka Panda | RGB + 深度 | 语言指令        | 86 种    | —           | 564    | 76 K      |
| OXE           | 22 种平台    | RGB + 深度 | 语言指令        | 527 种   | > 16 万对象 | 311    | > 1 000 K |