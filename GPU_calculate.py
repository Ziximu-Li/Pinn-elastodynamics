import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS
import random

seed = 111111
E = 2.1 * 1e9  # 杨氏模量
beta = 1.0 * 1e8
mu = 0.3  # 泊松比

# 设置随机种子
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

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义Softplus激活函数
class Softplus(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)

# 定义神经网络结构
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 20)  # 输入层到隐藏层
        self.fc2 = nn.Linear(20, 20)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(20, 20)  # 隐藏层到隐藏层
        self.fc4 = nn.Linear(20, 40)  # 隐藏层到隐藏层
        self.fc5 = nn.Linear(40, 40)  # 隐藏层到隐藏层
        self.fc6 = nn.Linear(40, 5)  # 隐藏层到输出层，输出5个分量：两个位移分量和三个应力分量

        self.swish = Swish()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = Softplus()

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.tanh(self.fc1(x))  # 激活函数
        x = self.tanh(self.fc2(x))  # 激活函数
        x = self.swish(self.fc3(x))  # 激活函数
        x = self.swish(self.fc4(x))  # 激活函数
        x = self.swish(self.fc5(x))  # 激活函数
        x = self.fc6(x)  # 输出层

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
    # data_normalized[:, 0] = data_normalized[:, 0] / 10 - 1.0
    # data_normalized[:, 1] = data_normalized[:, 1] / 2.5 - 1.0
    data_normalized[:, 0] = data_normalized[:, 0] - 10
    data_normalized[:, 1] = data_normalized[:, 1] - 2.5

    return data_normalized

def generate_BC_training_data(length, height, step):

    # 固定端边界条件 (x = 0)    length = 20, height = 5, step = 200
    x_boundary_fixed = torch.zeros((step, 2), device=device)  # 边界坐标点
    x_boundary_fixed[:, 1] = torch.linspace(0, height, step, device=device)  # 生成 x=0 边界的坐标  --> [0, y]T
    x_boundary_fixed.requires_grad_(True)
    y_boundary_fixed = torch.zeros((step, 2), device=device)  # 设置固定端的位移: y_boundary_fixed --> U and V (x = 0) = 0

    # 其他边界条件
    x_boundary = torch.cat([
        torch.cat([torch.linspace(0, length, step, device=device)[1:step-1, None], torch.zeros((step-2, 1), device=device)], dim=1),  # 下边界
        torch.cat([torch.linspace(0, length, step, device=device)[1:step-1, None], height * torch.ones((step-2, 1), device=device)], dim=1),  # 上边界
        torch.cat([length * torch.ones((step, 1), device=device), torch.linspace(0, height, step, device=device)[:, None]], dim=1)  # 右边界
    ]) # 定义上下右侧边界：x_boundary --> [fy(y=0) + fy(y=5) + fx(x=20)] --> [200*3, 2]T
    x_boundary.requires_grad_(True)

    y_boundary = torch.zeros((step, 2), device=device)  # 应力载荷条件的大小
    y_boundary[:, 0] = torch.linspace(5, 0, step, device=device)[:step]
    # y_boundary[:step, 1] = 0.0  # 下边界 \(\sigma_{yy} = 0\)
    # y_boundary[step:2 * step, 1] = 0.0  # 上边界 \(\sigma_{yy} = 0\)
    # y_boundary[2 * step:, 0] = 0.0  # 右边界 \(\sigma_{xx} = 0\)

    y_boundary.requires_grad_(True)

    return x_boundary_fixed, y_boundary_fixed, x_boundary, y_boundary

def calculate_sigma(u_pred, v_pred, x_interior):
    # 几何方程
    u_x = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(v_pred, x_interior, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v_pred, x_interior, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    epsilon_xx = u_x
    epsilon_yy = v_y
    epsilon_xy = u_y + v_x

    # 本构方程 (线弹性材料，杨氏模量 E 和泊松比 nu)

    C11 = E / (1 - mu ** 2)

    sigma_xx = C11 * (epsilon_xx + mu * epsilon_yy)
    sigma_yy = C11 * (mu * epsilon_xx + epsilon_yy)
    sigma_xy = C11 * ((1 + mu) * epsilon_xy / 2)

    sigma_xx.requires_grad_(True)
    sigma_xy.requires_grad_(True)
    sigma_yy.requires_grad_(True)

    return sigma_xx, sigma_yy, sigma_xy

def calculate_f(sigma_xx, sigma_yy, sigma_xy, x_interior):
    fxx_x = torch.autograd.grad(sigma_xx, x_interior,
                                grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0][:, 0]
    fxy_x = torch.autograd.grad(sigma_xy, x_interior,
                                grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 0]
    fxy_y = torch.autograd.grad(sigma_xy, x_interior,
                                grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 1]
    fyy_y = torch.autograd.grad(sigma_yy, x_interior,
                                grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0][:, 1]

    return fxx_x, fxy_x, fxy_y, fyy_y

# 定义损失函数，包含PDE损失和边界条件损失
def loss_fn(model, step, x_interior, x_boundary_fixed, y_boundary_fixed, x_boundary, y_boundary):

    # 将边界点加入内部点中，保证边界点同样满足弹性力学本构方程
    x_interior_loss_total = torch.cat((x_interior, x_boundary_fixed, x_boundary), dim=0)
    x_interior_loss_total.requires_grad_(True)

    # 内部点+上下边界点 满足平衡方程 f=0
    x_interior_without_F = torch.cat((x_interior, x_boundary[:2*step-4,]), dim=0)
    x_interior_without_F.requires_grad_(True)

    # 右侧边界点 施加载荷约束
    x_interior_right = x_boundary[2*step-4:,]
    x_interior_right.requires_grad_(True)

    # 上下边界点，边界应力为0
    x_boundary_up_down = x_boundary[: 2*step-4,]
    x_boundary_up_down.requires_grad_(True)

    # 计算PDE损失
    x_interior_loss_total = normalize_data(x_interior_loss_total)
    pred = model(x_interior_loss_total)
    u_pred, v_pred = pred[:, 0], pred[:, 1]
    sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = pred[:, 2], pred[:, 3], pred[:, 4]

    sigma_xx, sigma_yy, sigma_xy = calculate_sigma(u_pred, v_pred, x_interior_loss_total)

    # 本构方程
    phy_loss = (torch.mean((sigma_xx_pred - sigma_xx) ** 2) + \
                        torch.mean((sigma_xy_pred - sigma_xy) ** 2) + \
                        torch.mean((sigma_yy_pred - sigma_yy) ** 2))

    # 平衡方程损失
    x_interior_without_F = normalize_data(x_interior_without_F)
    pred_without_F = model(x_interior_without_F)
    u_pred_without_F, v_pred_without_F = pred_without_F[:, 0], pred_without_F[:, 1]
    sigma_xx_pred_without_F, sigma_yy_pred_without_F, sigma_xy_pred_without_F \
        = pred_without_F[:, 2], pred_without_F[:, 3], pred_without_F[:, 4]

    fxx_x, fxy_x, fxy_y, fyy_y = calculate_f(sigma_xx_pred_without_F, sigma_yy_pred_without_F,
                                             sigma_xy_pred_without_F, x_interior_without_F)

    balence_without_F_loss = (torch.mean((fxx_x + fxy_y) ** 2) + torch.mean((fxy_x + fyy_y) ** 2))

    #  右端载荷损失
    x_interior_right = normalize_data(x_interior_right)
    pred_right = model(x_interior_right)

    sigma_xx_pred_right, sigma_yy_pred_right, sigma_xy_pred_right \
        = pred_right[:, 2], pred_right[:, 3], pred_right[:, 4]

    # fxx_x_right, fxy_x_right, fxy_y_right, fyy_y_right\
    #     = calculate_f(sigma_xx_pred_right, sigma_yy_pred_right,
    #                   sigma_xy_pred_right, x_interior_right)

    balence_F_loss = (torch.mean((sigma_xx_pred_right - y_boundary[:, 0]) ** 2) +
                      torch.mean((sigma_yy_pred_right - y_boundary[:, 1]) ** 2) +
                      torch.mean((sigma_xy_pred_right - y_boundary[:, 1]) ** 2))

    # 上下端应力损失
    x_boundary_up_down = normalize_data(x_boundary_up_down)
    pred_up_down = model(x_boundary_up_down)

    sigma_xx_pred_up_down, sigma_yy_pred_up_down, sigma_xy_pred_up_down \
        = pred_up_down[:, 2], pred_up_down[:, 3], pred_up_down[:, 4]

    balence_up_down_F_loss = (torch.mean((sigma_xx_pred_up_down) ** 2) +
                              torch.mean((sigma_yy_pred_up_down) ** 2) +
                              torch.mean((sigma_xy_pred_up_down) ** 2))

    # 固定端条件损失
    x_boundary_fixed = normalize_data(x_boundary_fixed)
    fixed_pred = model(x_boundary_fixed)
    fixed_pred.requires_grad_(True)

    # u_pred_fixed = normalize_to_01(fixed_pred[:, 0])
    # v_pred_fixed = normalize_to_01(fixed_pred[:, 1])
    u_pred_fixed = fixed_pred[:, 0] * beta * 10
    v_pred_fixed = fixed_pred[:, 1] * beta * 10

    fixed_loss = (torch.mean((u_pred_fixed) ** 2) +
                  torch.mean((v_pred_fixed) ** 2))

    # fixed_fxx_x, fixed_fxy_x, fixed_fxy_y, fixed_fyy_y \
    #     = calculate_f(fixed_pred[:, 2], fixed_pred[:, 3],
    #                   fixed_pred[:, 4], x_boundary_fixed)
    #
    # loss_fixed_f = (torch.abs(torch.sum(fixed_fxy_x + fixed_fyy_y) - torch.sum(y_boundary[:, 1])) +
    #                 torch.abs(torch.sum(fixed_fxx_x + fixed_fxy_y) - torch.sum(y_boundary[:, 0])))

    # 强制性使得横向位移为正
    uv_loss = (torch.sum(torch.clamp(u_pred * beta, max=0.0) ** 2))

    lambda_balence = 1  # 平衡方程条件权重
    lambda_fixed = 1  # 固定端边界条件权重
    lambda_BC = 1  # 应力边界条件权重
    lambda_phy = 1  # 本构条件权重
    lambda_uv = 1

    PINN_loss = (lambda_balence * balence_without_F_loss +
                 lambda_fixed * fixed_loss +
                 lambda_BC * (balence_F_loss + balence_up_down_F_loss) +
                 lambda_phy * phy_loss +
                 lambda_uv * uv_loss)

    return (PINN_loss, balence_without_F_loss, fixed_loss, balence_F_loss + balence_up_down_F_loss, phy_loss, uv_loss)

# 主训练过程
def train(maxiters, n, num_phi_train, step):

    model = PINN().to(device)  # 初始化PINN模型并移动到GPU
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器
    loss_history = []

    steps_count = maxiters // n  # 每次加点训练的次数
    num_print_epoch = 1000

    # 悬臂梁尺寸
    length = 20.0
    height = 5.0

    # 边界点取点数
    x_boundary_fixed, y_boundary_fixed, x_boundary, y_boundary = (
        generate_BC_training_data(length, height, step))
    x_boundary_fixed.requires_grad_(True), y_boundary_fixed.requires_grad_(True)
    x_boundary.requires_grad_(True), y_boundary.requires_grad_(True)

    # 初始化空的内部点
    x_interior_total = torch.empty(0, 2, device=device)

    for i in range(n):
        train_interior = int(num_phi_train / n)
        step_train_start_time = time.time()

        if i < n :
            x_interior = torch.rand((train_interior, 2), device=device)  # 随机生成内部点
            x_interior[:, 0] *= length
            x_interior[:, 1] *= height
            x_interior.requires_grad_(True)

            x_interior_total = torch.cat((x_interior_total, x_interior), dim=0)  # 将新生成的点加入总的内部点集合
            x_interior_total.requires_grad_(True)

            for epoch in range(steps_count):
                optimizer.zero_grad()
                loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, uv_loss\
                    = loss_fn(model, step, x_interior_total,
                              x_boundary_fixed, y_boundary_fixed,
                              x_boundary, y_boundary)
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                loss_history.append(loss.item())

                if epoch % num_print_epoch == 0:
                    print(f'Iteration {i + 1}/{n}, Epoch {epoch}, '
                          f'Loss: {loss.item():.6f}, '
                          f'balence_without_F_loss: {balence_without_F_loss.item():.6f}, '
                          f'fixed_loss: {fixed_loss.item():.6f}, '
                          f'balence_F_loss: {balence_F_loss.item():.6f}, '
                          f'phy_loss: {phy_loss.item():.6f}, '
                          f'uv_loss: {uv_loss.item():.6f} ')

        else:
            final_interior = num_phi_train - (n - 1) * train_interior
            x_interior = torch.rand((final_interior, 2), device=device)  # 随机生成内部点
            x_interior[:, 0] *= length
            x_interior[:, 1] *= height
            x_interior.requires_grad_(True)

            x_interior_total = torch.cat((x_interior_total, x_interior), dim=0)  # 将新生成的点加入总的内部点集合
            x_interior_total.requires_grad_(True)

            optimizer = LBFGS(model.parameters(), lr=0.001, max_iter=1000, history_size=50, line_search_fn='strong_wolfe')  # 最后一轮使用LBFGS优化器

            def closure():
                optimizer.zero_grad()
                loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, sigma_loss \
                    = loss_fn(model, step, x_interior_total,
                              x_boundary_fixed, y_boundary_fixed,
                              x_boundary, y_boundary)
                loss.backward(retain_graph=True)
                return loss

            for epoch in range(int(steps_count / 2)):
                loss = optimizer.step(closure)  # LBFGS优化

                loss_history.append(loss.item())

                if epoch % num_print_epoch == 0:
                    print(f'Iteration {i + 1}/{n}, Epoch {epoch}, Loss: {loss.item():.5f}')

        step_train_time = time.time() - step_train_start_time
        print(f'Step {i + 1}/{n}, Time Elapsed: {step_train_time:.2f}s')

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

    fxx_x = torch.autograd.grad(sigma_xx, xy_tensor,
                                grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0][:, 0]
    fxy_x = torch.autograd.grad(sigma_xy, xy_tensor,
                                grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 0]
    fxy_y = torch.autograd.grad(sigma_xy, xy_tensor,
                                grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 1]
    fyy_y = torch.autograd.grad(sigma_yy, xy_tensor,
                                grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0][:, 1]

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

    fxx_x_calculate = torch.autograd.grad(calculate_sigma_xx, xy_tensor,
                                grad_outputs=torch.ones_like(calculate_sigma_xx), create_graph=True)[0][:, 0]
    fxy_x_calculate = torch.autograd.grad(calculate_sigma_xy, xy_tensor,
                                grad_outputs=torch.ones_like(calculate_sigma_xy), create_graph=True)[0][:, 0]
    fxy_y_calculate = torch.autograd.grad(calculate_sigma_xy, xy_tensor,
                                grad_outputs=torch.ones_like(calculate_sigma_xy), create_graph=True)[0][:, 1]
    fyy_y_calculate = torch.autograd.grad(calculate_sigma_yy, xy_tensor,
                                grad_outputs=torch.ones_like(calculate_sigma_yy), create_graph=True)[0][:, 1]

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

    fxx_x_calculate = fxx_x_calculate.reshape(1000, 1000).detach().cpu().numpy()
    fxy_x_calculate = fxy_x_calculate.reshape(1000, 1000).detach().cpu().numpy()
    fxy_y_calculate = fxy_y_calculate.reshape(1000, 1000).detach().cpu().numpy()
    fyy_y_calculate = fyy_y_calculate.reshape(1000, 1000).detach().cpu().numpy()


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

    # 绘制平衡方程云图
    plt.figure()
    plt.imshow(fxx_x + fxy_y, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('fxx_x + fxy_y')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    plt.figure()
    plt.imshow(fxy_x + fyy_y, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('fxy_x + fyy_y')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    plt.figure()
    plt.imshow(fxx_x_calculate + fxy_y_calculate, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('fxx_x_calculate + fxy_y_calculate')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    plt.figure()
    plt.imshow(fxy_x_calculate + fyy_y_calculate, extent=(0, 20, 0, 5), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('fxy_x_calculate + fyy_y_calculate')
    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.show()

    # 绘制损失函数曲线
    plt.figure()
    plt.plot(loss_history[:])
    plt.title('Loss Function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    num_phi_train = 1000  # 总加点个数
    n = 1  # 加点轮次
    maxiters = 30000  # 总共训练的次数
    step = 100  # 定义边界加点个数

    start_time = time.time()
    model, loss_history = train(maxiters, n, num_phi_train, step)  # 传递step参数
    elapsed_time = time.time() - start_time
    print(f'Time Elapsed: {elapsed_time:.2f}s')
    plot_results(model, loss_history)
