from lerobot.common.policies import make_policy

# 示例：创建一个Diffusion Policy实例
diffusion_policy_config = {
    "policy_type": "diffusion",
    # 特定于Diffusion Policy的参数
    "obs_dim": 256, # 假设观察特征维度
    "action_dim": 7, # 动作维度
    "pred_horizon": 16, # 预测视野
    "obs_horizon": 2,   # 观察视野
    "action_horizon": 8, # 动作视野
    # ... 其他Diffusion Policy参数
}
policy_instance = make_policy(
    policy_name_or_path=diffusion_policy_config["policy_type"], # 可以是类型名或HF Hub路径
    # checkpoint_path=None, # 可选，加载预训练权重的路径
    # device="cuda" if torch.cuda.is_available() else "cpu",
    **diffusion_policy_config # 将配置字典解包作为参数
)

print(f"成功创建策略: {type(policy_instance)}")