import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS
import random

seed = 12345
E = 2.1 * 1e9  # 杨氏模量
beta = 1 * 1e8
mu = 0.3  # 泊松比

# 悬臂梁尺寸
length = 20.0
height = 5.0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)
device = torch.device("cuda")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class Softplus(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 60)  # 输入层到隐藏层
        self.fc2 = nn.Linear(60, 60)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(60, 60)  # 隐藏层到隐藏层
        self.fc4 = nn.Linear(60, 60)  # 隐藏层到隐藏层
        self.fc5 = nn.Linear(60, 60)  # 隐藏层到隐藏层
        self.fc6 = nn.Linear(60, 5)  # 隐藏层到输出层，输出5个分量：两个位移分量和三个应力分量

        self.swish = Swish()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = Softplus()

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.softplus(self.fc1(x))  # 激活函数
        x = self.softplus(self.fc2(x))  # 激活函数
        x = self.softplus(self.fc3(x))  # 激活函数
        x = self.softplus(self.fc4(x))  # 激活函数
        x = self.softplus(self.fc5(x))  # 激活函数
        x = self.fc6(x)  # 输出层

        # 输出进行一个线性激活
        x[:, 0] = x[:, 0] / beta
        x[:, 1] = x[:, 1] / beta

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 使用Xavier初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始化偏置为0

def normalize_data(data):
    data_normalized = data.clone()  # 先克隆数据，避免在原始数据上进行in-place操作
    data_normalized[:, 0] = data_normalized[:, 0] - 10
    data_normalized[:, 1] = data_normalized[:, 1] - 2.5
    return data_normalized

def calculate_sigma(u_pred, v_pred, x_interior):
    # 几何方程
    u_x = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(v_pred, x_interior, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v_pred, x_interior, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    epsilon_xx = u_x
    epsilon_yy = v_y
    epsilon_xy = u_y + v_x

    # 本构方程 (线弹性材料，杨氏模量 E 和泊松比 mu)
    C11 = E / (1 - mu ** 2)

    sigma_xx = C11 * (epsilon_xx + mu * epsilon_yy)
    sigma_yy = C11 * (mu * epsilon_xx + epsilon_yy)
    sigma_xy = C11 * ((1 + mu) * epsilon_xy / 2)

    sigma_xx.requires_grad_(True)
    sigma_xy.requires_grad_(True)
    sigma_yy.requires_grad_(True)

    return sigma_xx, sigma_yy, sigma_xy

def calculate_f(sigma_xx, sigma_yy, sigma_xy, x_interior):
    fxx_x = torch.autograd.grad(sigma_xx, x_interior, grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0][:, 0]
    fxy_x = torch.autograd.grad(sigma_xy, x_interior, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 0]
    fxy_y = torch.autograd.grad(sigma_xy, x_interior, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 1]
    fyy_y = torch.autograd.grad(sigma_yy, x_interior, grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0][:, 1]

    return fxx_x, fxy_x, fxy_y, fyy_y

# 定义损失函数，包含PDE损失和边界条件损失
def loss_fn(model, step, step_bc, inside, boundary, boundary_stress):
    # 各边数据点个数
    length_point = int(length / step)
    height_point = int(height / step)
    inside_point = (length_point - 1) * (height_point - 1)
    length_point = int(length / step_bc)
    height_point = int(height / step_bc)
    left_righr_point = (height_point + 1) * 2
    up_down_point = (length_point - 1) * 2

    x_interior = torch.cat((inside, boundary), dim=0)
    x_interior.requires_grad_(True)
    x_interior = normalize_data(x_interior)

    # 计算损失
    pred = model(x_interior)
    pred.requires_grad_(True)
    pred = pred.to(device)

    u_pred, v_pred = pred[:, 0], pred[:, 1]
    sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = pred[:, 2], pred[:, 3], pred[:, 4]

    sigma_xx, sigma_yy, sigma_xy = calculate_sigma(u_pred, v_pred, x_interior)
    fxx_x_pred, fxy_x_pred, fxy_y_pred, fyy_y_pred = calculate_f(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, x_interior)
    # fxx_x_pred, fxy_x_pred, fxy_y_pred, fyy_y_pred = calculate_f(sigma_xx, sigma_yy, sigma_xy, x_interior)

    # 本构损失
    phy_loss = (torch.mean((sigma_xx_pred - sigma_xx) ** 2) +
                torch.mean((sigma_xy_pred - sigma_xy) ** 2) +
                torch.mean((sigma_yy_pred - sigma_yy) ** 2))

    # 平衡方程损失
    up_point = inside_point + (height_point + 1)
    right_point = inside_point + (height_point + 1) + up_down_point

    fxx_x = torch.cat((fxx_x_pred[:inside_point, ], fxx_x_pred[up_point: right_point, ]), dim=0)
    fxy_x = torch.cat((fxy_x_pred[:inside_point, ], fxy_x_pred[up_point: right_point, ]), dim=0)
    fxy_y = torch.cat((fxy_y_pred[:inside_point, ], fxy_y_pred[up_point: right_point, ]), dim=0)
    fyy_y = torch.cat((fyy_y_pred[:inside_point, ], fyy_y_pred[up_point: right_point, ]), dim=0)

    balence_without_F_loss = (torch.mean((fxx_x + fxy_y) ** 2) + torch.mean((fxy_x + fyy_y) ** 2))

    # 右端载荷损失
    pred_right = pred[right_point:,]
    sigma_xx_pred_right, sigma_yy_pred_right, sigma_xy_pred_right = pred_right[:, 2], pred_right[:, 3], pred_right[:, 4]

    balence_F_loss = (torch.mean((sigma_xx_pred_right - boundary_stress) ** 2) +
                      torch.mean(sigma_yy_pred_right ** 2) +
                      torch.mean(sigma_xy_pred_right ** 2))

    # 上下端应力损失
    pred_up_down = pred[up_point: right_point,]
    sigma_xx_pred_up_down, sigma_yy_pred_up_down, sigma_xy_pred_up_down = pred_up_down[:, 2], pred_up_down[:, 3], pred_up_down[:, 4]
    balence_up_down_F_loss = (torch.mean((sigma_yy_pred_up_down) ** 2) + torch.mean((sigma_xy_pred_up_down) ** 2))

    # 固定端条件损失
    fixed_pred = pred[inside_point: up_point,]

    u_pred_fixed = fixed_pred[:, 0] * beta
    v_pred_fixed = fixed_pred[:, 1] * beta

    fixed_loss = (torch.mean((u_pred_fixed) ** 2) + torch.mean((v_pred_fixed) ** 2))

    # 强制性使得位移为正
    uv_loss = (torch.sum(torch.clamp(u_pred * beta, max=0.0) ** 2) + torch.sum(
        torch.clamp(v_pred * beta, max=0.0) ** 2))

    lambda_balence = 1  # 平衡方程条件权重
    lambda_fixed = 1  # 固定端边界条件权重
    lambda_BC = 1  # 应力边界条件权重
    lambda_phy = 1  # 本构条件权重
    lambda_uv = 1  # 强制横向位移为正

    PINN_loss = (lambda_balence * balence_without_F_loss +
                 lambda_fixed * fixed_loss +
                 lambda_BC * (balence_F_loss + balence_up_down_F_loss) +
                 lambda_phy * phy_loss +
                 lambda_uv * uv_loss)

    return (PINN_loss, balence_without_F_loss, fixed_loss, balence_F_loss + balence_up_down_F_loss, phy_loss, uv_loss)

# 主训练过程
def train(maxiters, adamsnum, step, step_bc):
    model = PINN().to(device)  # 初始化PINN模型并移动到GPU
    optimizer = optim.Adam(model.parameters(),lr=0.01)  # 使用Adam优化器
    loss_history = []

    # -------------------------- 设置训练集 -------------------------- #
    x = torch.arange(step, length, step)
    y = torch.arange(step, height, step)

    # 内部点坐标
    inside = torch.stack(torch.meshgrid(x, y, indexing='ij')).reshape(2, -1).T

    y = torch.arange(0, height + step_bc, step_bc)
    x = torch.arange(step_bc, length, step_bc)
    # 上侧
    boundary_up = torch.stack(torch.meshgrid(x, y[-1], indexing='ij')).reshape(2, -1).T
    # 下侧
    boundary_down = torch.stack(torch.meshgrid(x, y[0], indexing='ij')).reshape(2, -1).T

    x = torch.arange(0, length + step_bc, step_bc)
    # 左侧
    boundary_left = torch.stack(torch.meshgrid(x[0], y, indexing='ij')).reshape(2, -1).T
    # 右侧
    boundary_right = torch.stack(torch.meshgrid(x[-1], y, indexing='ij')).reshape(2, -1).T
    # 整合
    boundary = torch.cat([boundary_left, boundary_down, boundary_up, boundary_right])

    # 右侧边界点数值
    boundary_stress = torch.arange(height, -step_bc, -step_bc)

    # 数据处理
    inside.requires_grad_(True)
    inside = inside.to(device)
    boundary.requires_grad_(True)
    boundary = boundary.to(device)
    boundary_stress = boundary_stress.to(device)
    boundary_stress.requires_grad_(True)
    # -------------------------- 数据集设置完成 -------------------------- #

    step_start_time = time.time()
    adams_start_time = time.time()

    for epoch in range(maxiters):
        if epoch < adamsnum + 1:
            optimizer.zero_grad()
            loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, uv_loss = \
                loss_fn(model, step, step_bc, inside, boundary, boundary_stress)
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            loss_history.append(loss.item())

            if epoch % 200 == 0:
                step_train_time = time.time() - step_start_time
                print(f'Epoch {epoch}, '
                      f'Loss: {loss.item():.6f}, '
                      f'balence_without_F_loss: {balence_without_F_loss.item():.6f}, '
                      f'fixed_loss: {fixed_loss.item():.6f}, '
                      f'balence_F_loss: {balence_F_loss.item():.6f}, '
                      f'phy_loss: {phy_loss.item():.6f}, '
                      f'uv_loss: {uv_loss.item():.6f}，'
                      f'Step_Time: {step_train_time:.2f}s')
                step_start_time = time.time()

            if epoch == adamsnum:
                adams_time = time.time() - adams_start_time
                print(f'Adams time: {adams_time:.2f}s')
        else:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0,
                                          max_iter=1000,
                                          history_size=10,
                                          tolerance_grad=0.00001 * np.finfo(float).eps)

            def closure():
                optimizer.zero_grad()
                loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, uv_loss = \
                    loss_fn(model, step, step_bc, inside, boundary, boundary_stress)
                loss.backward()
                return loss

            loss = optimizer.step(closure)  # LBFGS优化

            loss_history.append(loss.item())

            if epoch % 10 == 0:
                step_train_time = time.time() - step_start_time
                print(f'Epoch {epoch}, '
                      f'Loss: {loss.item():.6f}, '
                      f'Step_Time: {step_train_time:.2f}s')
                step_start_time = time.time()

    return model, loss_history

# 绘制最终应力和位移云图
def plot_results(model, loss_history):
    x = np.linspace(0, 20, 1000)
    y = np.linspace(0, 5, 1000)
    X, Y = np.meshgrid(x, y)

    xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True).to(device)
    xy_tensor = normalize_data(xy_tensor)
    pred = model(xy_tensor)
    U = pred[:, 0]
    V = pred[:, 1]
    sigma_xx = pred[:, 2]
    sigma_yy = pred[:, 3]
    sigma_xy = pred[:, 4]

    fxx_x = torch.autograd.grad(sigma_xx, xy_tensor, grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0][:, 0]
    fxy_x = torch.autograd.grad(sigma_xy, xy_tensor, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 0]
    fxy_y = torch.autograd.grad(sigma_xy, xy_tensor, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 1]
    fyy_y = torch.autograd.grad(sigma_yy, xy_tensor, grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0][:, 1]

    # 几何方程
    u_x = torch.autograd.grad(U, xy_tensor, grad_outputs=torch.ones_like(pred[:, 0]), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(U, xy_tensor, grad_outputs=torch.ones_like(pred[:, 0]), create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(V, xy_tensor, grad_outputs=torch.ones_like(pred[:, 1]), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(V, xy_tensor, grad_outputs=torch.ones_like(pred[:, 1]), create_graph=True)[0][:, 1]

    epsilon_xx = u_x
    epsilon_yy = v_y
    epsilon_xy = u_y + v_x

    C11 = E / (1 - mu ** 2)
    C12 = mu * E / (1 - mu ** 2)
    C33 = E / (2 * (1 + mu))

    calculate_sigma_xx = (C11 * epsilon_xx + C12 * epsilon_yy)
    calculate_sigma_yy = (C12 * epsilon_xx + C11 * epsilon_yy)
    calculate_sigma_xy = (C33 * epsilon_xy)

    U = pred[:, 0].reshape(1000, 1000).detach().cpu().numpy()
    V = pred[:, 1].reshape(1000, 1000).detach().cpu().numpy()
    sigma_xx = pred[:, 2].reshape(1000, 1000).detach().cpu().numpy()
    sigma_yy = pred[:, 3].reshape(1000, 1000).detach().cpu().numpy()
    sigma_xy = pred[:, 4].reshape(1000, 1000).detach().cpu().numpy()

    calculate_sigma_xx = calculate_sigma_xx.reshape(1000, 1000).detach().cpu().numpy()
    calculate_sigma_yy = calculate_sigma_yy.reshape(1000, 1000).detach().cpu().numpy()
    calculate_sigma_xy = calculate_sigma_xy.reshape(1000, 1000).detach().cpu().numpy()

    fxx_x = fxx_x.reshape(1000, 1000).detach().cpu().numpy()
    fxy_x = fxy_x.reshape(1000, 1000).detach().cpu().numpy()
    fxy_y = fxy_y.reshape(1000, 1000).detach().cpu().numpy()
    fyy_y = fyy_y.reshape(1000, 1000).detach().cpu().numpy()

    # 绘制位移云图 U
    plt.figure()
    plt.imshow(U, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Displacement U')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    # 绘制位移云图 V
    plt.figure()
    plt.imshow(V, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Displacement V')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    # 绘制应力云图 sigma_xx
    plt.figure()
    plt.imshow(sigma_xx, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Stress $\sigma_{xx}$')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    # 绘制应力云图 sigma_yy
    plt.figure()
    plt.imshow(sigma_yy, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Stress $\sigma_{yy}$')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    # 绘制应力云图 sigma_xy
    plt.figure()
    plt.imshow(sigma_xy, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Stress $\sigma_{xy}$')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    # 绘制损失函数曲线
    plt.figure()
    plt.plot(loss_history[500:])
    plt.title('Loss Function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    maxiters = 600  # 总共训练的次数
    step = 0.25  # 定义内部加点
    step_bc = 0.1  # 定义边界加点
    adamsnum = 500

    start_time = time.time()
    model, loss_history = train(maxiters, adamsnum, step, step_bc)

    elapsed_time = time.time() - start_time
    print(f'Time Elapsed: {elapsed_time:.2f}s')
    plot_results(model, loss_history)
