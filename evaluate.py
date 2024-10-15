import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pandas as pd

# 定义PINN类以及其前向传播和权重初始化方式
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Softplus(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 5)
        self.softplus = Softplus()
        self._initialize_weights()

    def forward(self, x):
        x = self.softplus(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        x = self.softplus(self.fc4(x))
        x = self.softplus(self.fc5(x))
        x = self.fc6(x)
        x[:, 0] = x[:, 0] / beta
        x[:, 1] = x[:, 1] / beta
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# 选择GPU或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E = 2.1 * 1e9  # 杨氏模量
beta = 1 * 1e8
mu = 0.3  # 泊松比

# 从文件加载已经训练完成的模型权重
model_loaded = torch.load('../5outputModel.pth', map_location=device)
model_loaded.eval()  # 设置模型为evaluation状态

def normalize_data(data):
    data_normalized = data.clone()
    data_normalized[:, 0] = data_normalized[:, 0] - 10
    data_normalized[:, 1] = data_normalized[:, 1] - 2.5
    return data_normalized

# 生成时空网格
length = 20.0
height = 5.0
step = 0.1
x = np.arange(0, length + step, step)  # x方向取点
y = np.arange(0, height + step, step)   # y方向取点
X, Y = np.meshgrid(x, y)
length_step = int(length / step + 1)
height_step = int(height / step + 1)

xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True).to(device)

xy_tensor = normalize_data(xy_tensor)
pred = model_loaded(xy_tensor)

U = pred[:, 0]
V = pred[:, 1]
pred_S11 = pred[:, 2]
pred_S22 = pred[:, 3]
pred_S12 = pred[:, 4]

U = pred[:, 0].reshape(height_step, length_step).detach().cpu().numpy()
V = pred[:, 1].reshape(height_step, length_step).detach().cpu().numpy()
pred_S11 = pred[:, 2].reshape(height_step, length_step).detach().cpu().numpy()
pred_S12 = pred[:, 4].reshape(height_step, length_step).detach().cpu().numpy()
pred_S22 = pred[:, 3].reshape(height_step, length_step).detach().cpu().numpy()

# ---------------------------------- 获取abaqus结点值 ---------------------------------- #
# 读取节点数据
abaqus_data = pd.read_csv('node_stress_data.csv')

# 提取ABAQUS中的横向位移 (U1) 和纵向位移 (U2)
X_abaqus = abaqus_data['X'].values  # 提取X坐标
Y_abaqus = abaqus_data['Y'].values  # 提取Y坐标
U_abaqus = abaqus_data['U1'].values  # 横向位移 U1
V_abaqus = abaqus_data['U2'].values  # 纵向位移 U2

# 将ABAQUS的X, Y坐标调整为与PINN相同的形状
X_abaqus = X_abaqus.reshape(height_step, length_step)
Y_abaqus = Y_abaqus.reshape(height_step, length_step)

# 将ABAQUS的位移分量U1, U2调整为与PINN相同的形状
U_abaqus = U_abaqus.reshape(height_step, length_step)
V_abaqus = V_abaqus.reshape(height_step, length_step)

# 计算横向位移 (U) 和纵向位移 (V) 的相对误差
epsilon = 5e-8
U_relative_error = np.abs((U_abaqus - U) / (np.abs(U_abaqus) + epsilon))  # 横向位移 U 的相对误差
V_relative_error = np.abs((V_abaqus - V) / (np.abs(V_abaqus) + epsilon))  # 纵向位移 V 的相对误差

# 绘制横向位移 U 的相对误差分布
plt.figure()
plt.title('Relative Error in U (Displacement X)')
plt.imshow(U_relative_error, extent=(0, length, 0, height), origin='lower', cmap='jet')
plt.colorbar(label='Relative Error in U')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 绘制纵向位移 V 的相对误差分布
plt.figure()
plt.title('Relative Error in V (Displacement Y)')
plt.imshow(V_relative_error, extent=(0, length, 0, height), origin='lower', cmap='jet')
plt.colorbar(label='Relative Error in V')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 打印相对误差的最大值
print("Max relative error in U:", np.max(U_relative_error))
print("Max relative error in V:", np.max(V_relative_error))

# 提取ABAQUS插值后的应力数据
S11_abaqus = abaqus_data['S11'].values  # 从ABAQUS插值后的S11应力
S22_abaqus = abaqus_data['S22'].values  # 从ABAQUS插值后的S22应力
S12_abaqus = abaqus_data['S12'].values  # 从ABAQUS插值后的S12应力

# 将应力数据调整为正确的形状
S11_abaqus = S11_abaqus.reshape(height_step, length_step)
S22_abaqus = S22_abaqus.reshape(height_step, length_step)
S12_abaqus = S12_abaqus.reshape(height_step, length_step)

# 计算 S11, S22, S12 的相对误差
epsilon = 5.0
S11_relative_error = np.abs((S11_abaqus - pred_S11) / (np.abs(S11_abaqus) + epsilon))
S22_relative_error = np.abs((S22_abaqus - pred_S22) / (np.abs(S22_abaqus) + epsilon))
S12_relative_error = np.abs((S12_abaqus - pred_S12) / (np.abs(S12_abaqus) + epsilon))

# 绘制 S11 相对误差云图
plt.figure()
plt.title('Relative Error in S11 (Stress X)')
plt.imshow(S11_relative_error, extent=(0, length, 0, height), origin='lower', cmap='jet')
plt.colorbar(label='Relative Error in S11')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 绘制 S22 相对误差云图
plt.figure()
plt.title('Relative Error in S22 (Stress Y)')
plt.imshow(S22_relative_error, extent=(0, length, 0, height), origin='lower', cmap='jet')
plt.colorbar(label='Relative Error in S22')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 绘制 S12 相对误差云图
plt.figure()
plt.title('Relative Error in S12 (Shear Stress)')
plt.imshow(S12_relative_error, extent=(0, length, 0, height), origin='lower', cmap='jet')
plt.colorbar(label='Relative Error in S12')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 打印最大相对误差
print("Max relative error in S11:", np.max(S11_relative_error))
print("Max relative error in S22:", np.max(S22_relative_error))
print("Max relative error in S12:", np.max(S12_relative_error))
