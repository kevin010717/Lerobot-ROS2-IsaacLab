# %%
# 讲义地址 https://www.cnblogs.com/zhangbo2008/p/17341284.html
import os
import io
import numpy as np
import torch
import torch.nn as nn

# ---- 重要：在导入 pyplot 之前设置无头后端 ----
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.datasets import make_s_curve

# -----------------------------
# 0) 设备与随机种子
# -----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 允许 cuDNN 选最快算法

# %%
# 1) 生成数据（在 CPU 上用 sklearn/NumPy 生成，随后按需搬到 GPU）
s_curve, _ = make_s_curve(10**4, noise=0.1, random_state=seed)  # 生成一个三维曲线.
s_curve = s_curve[:, [0, 2]] / 10.0  # 只取 x,y 轴.
print("shape of s:", np.shape(s_curve))

data_np = s_curve.T  # 仅用于首次可视化
fig, ax = plt.subplots()
ax.scatter(data_np[0], data_np[1], color='blue', edgecolors='white')
ax.axis('off')
plt.show()  # 在 Agg 后端下不会弹窗，无碍

# ---- 关键修正：数据集留在 CPU，便于 DataLoader pin_memory ----
dataset = torch.tensor(s_curve, dtype=torch.float32)  # 不要 .to(device)

# %%
# 2) 超参数
num_steps = 100

# 制定每一步的 beta（直接在目标设备上创建）
betas = torch.linspace(-6, 6, num_steps, device=device)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5  # 一堆 ~1e-5 的数

# 计算 alpha 系列变量（全部放在 device）
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1.0], device=device), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert (
    alphas.shape == alphas_prod.shape == alphas_prod_p.shape ==
    alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape ==
    one_minus_alphas_bar_sqrt.shape
)
print("all the same shape", betas.shape)

# %%
# 3) 任意时刻的采样
def q_x(x_0, t):
    """基于 x[0] 得到任意时刻 t 的 x[t]；x_0, 返回张量均在当前 device"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_1_m_t * noise

# %%
# 4) 展示若干加噪步
num_shows = 20
fig, axs = plt.subplots(2, 10, figsize=(28, 3))
plt.rc('text', color='black')

# 将 CPU 的 dataset 临时搬上 device 以便计算 q_x；也可直接用 CPU 版本运算
dataset_dev = dataset.to(device)
for i in range(num_shows):
    j = i // 10
    k = i % 10
    t_idx = torch.tensor([i * num_steps // num_shows], device=device)
    q_i = q_x(dataset_dev, t_idx)  # 在 device 上
    # Matplotlib 需要 CPU/NumPy
    x = q_i[:, 0].detach().cpu().numpy()
    y = q_i[:, 1].detach().cpu().numpy()
    axs[j, k].scatter(x, y, color='red', edgecolors='white')
    axs[j, k].set_axis_off()
    axs[j, k].set_title(f'$q(\\mathbf{{x}}_{{{int(t_idx.item())}}})$')

plt.show()

# %%
# 5) 逆扩散模型
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        # t: LongTensor [B] on same device
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)  # linear
            x = x + t_embedding
            x = self.linears[2 * idx + 1](x)  # relu
        x = self.linears[-1](x)
        return x

# %%
# 6) 训练的误差函数
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻 t 采样计算 loss"""
    batch_size = x_0.shape[0]

    # 生成随机时刻 t（保持在 device 上）
    t = torch.randint(0, n_steps, size=(batch_size // 2,), device=x_0.device)
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)  # [B,1]

    # 系数
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]

    # 随机噪声 eps
    e = torch.randn_like(x_0)

    # 构造模型输入
    x = x_0 * a + e * aml

    # 送入模型，得到噪声预测
    output = model(x, t.squeeze(-1).long())

    # 与真实噪声计算 MSE
    return (e - output).square().mean()

# %%
# 7) 逆扩散采样（inference）
@torch.no_grad()
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从 x[t] 采样到 x[t-1]"""
    # 保证标量 t 在同一设备
    t = torch.tensor([t], device=x.device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t.long())
    mean = (1 / (1 - betas[t]).sqrt()) * (x - coeff * eps_theta)
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

@torch.no_grad()
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, device):
    """从 x[T] 恢复 ... -> x[0]；返回每步样本列表（在 device 上）"""
    cur_x = torch.randn(shape, device=device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

# %%
# 8) 训练
print('Training model...')
batch_size = 128

# DataLoader 从 CPU 张量读取；CUDA 下 pin_memory=True 更快
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    pin_memory=(device.type == "cuda")
)

num_epoch = 4000
plt.rc('text', color='blue')

model = MLPDiffusion(num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x_seq = None  # 供后续动画使用
for t_epoch in range(num_epoch):
    for batch_x in dataloader:
        # DataLoader 产出在 CPU，上 GPU
        batch_x = batch_x.to(device, non_blocking=True)

        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    if (t_epoch % 100) == 0:
        print(f"[epoch {t_epoch}] loss = {loss.item():.6f}")
        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt, device)

        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach()
            xs = cur_x[:, 0].detach().cpu().numpy()
            ys = cur_x[:, 1].detach().cpu().numpy()
            axs[i - 1].scatter(xs, ys, color='red', edgecolors='white')
            axs[i - 1].set_axis_off()
            axs[i - 1].set_title(f'$q(\\mathbf{{x}}_{{{i*10}}})$')
        plt.show()

# %%
# 9) 动画演示扩散过程和逆扩散过程
imgs = []
for i in range(100):
    plt.clf()
    q_i = q_x(dataset_dev, torch.tensor([i], device=device))
    xs = q_i[:, 0].detach().cpu().numpy()
    ys = q_i[:, 1].detach().cpu().numpy()
    plt.scatter(xs, ys, color='red', edgecolors='white', s=5)
    plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)              # 关键：回到缓冲区起始位置
    img = Image.open(img_buf)
    imgs.append(img)

reverse = []
if x_seq is None:
    # 若训练循环未生成 x_seq，这里也可单独跑一遍采样用于演示
    x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt, device)

for i in range(100):
    plt.clf()
    cur_x = x_seq[i].detach()
    xs = cur_x[:, 0].detach().cpu().numpy()
    ys = cur_x[:, 1].detach().cpu().numpy()
    plt.scatter(xs, ys, color='red', edgecolors='white', s=5)
    plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)              # 关键：回到缓冲区起始位置
    img = Image.open(img_buf)
    reverse.append(img)

imgs = imgs + reverse
imgs[0].save("diffusion.gif", format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)
print("Saved diffusion.gif")
# %%
