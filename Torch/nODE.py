import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint


# 定义常微分方程
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

    def forward(self, t, y):
        return self.net(y)


# 定义神经ODE模型
class NeuralODE(nn.Module):
    def __init__(self, odefunc):
        super(NeuralODE, self).__init__()
        self.odefunc = odefunc

    def forward(self, y0, t):
        return odeint(self.odefunc, y0, t)


# 生成一些训练数据
def generate_spiral(n_samples, noise_std=0.1):
    t = torch.linspace(0, 4 * torch.pi, n_samples)
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    data = torch.stack([x, y], dim=1)
    data += noise_std * torch.randn_like(data)
    return data


# 模型训练
def train(model, data, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_y = model(data[0], t)
        loss = criterion(pred_y, data)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


# 超参数设置
n_samples = 100
noise_std = 0.1
t = torch.linspace(0, 1, n_samples)

# 初始化数据和模型
data = generate_spiral(n_samples, noise_std)
odefunc = ODEFunc()
model = NeuralODE(odefunc)

# 训练模型
train(model, data)
