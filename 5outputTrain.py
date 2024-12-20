import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS
import random

seed = 123456
E = 2.1 * 1e9  # 杨氏模量
mu = 0.3  # 泊松比
beta = 1.0 * 1e8

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
device = torch.device("cuda")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class Softplus(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)
class polynomial (nn.Module):
    def forward(self, x):
        data = x.clone()
        return (data + data**2)/10

# 定义一个类，用于实现PINN(Physics-informed Neural Networks)
class PINN(nn.Module):
    # 构造函数
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
        self.poly = polynomial()

        # 初始化权重
        self._initialize_weights()
        # 移动模型到 GPU
        self.to(device)  # 在初始化时，将整个模型移动到 GPU

        # 悬臂梁尺寸
        self.length = 20.0
        self.height = 5.0

        # 边界点取点数
        self.step = 0.25  # 定义内部加点
        self.step_bc = 0.1  # 定义边界加点

        # self.maxiters = 15000
        # self.adamsnum = 10000
        self.maxiters = 10000
        self.adamsnum = 5000

        self.step_start_time = time.time()
        self.adams_start_time = time.time()

        # -------------------------- 设置训练集 -------------------------- #
        x = torch.arange(self.step, self.length, self.step)
        y = torch.arange(self.step, self.height, self.step)

        # 内部点坐标
        self.inside = torch.stack(torch.meshgrid(x, y, indexing='ij')).reshape(2, -1).T

        y = torch.arange(0, self.height + self.step_bc, self.step_bc)
        x = torch.arange(self.step_bc, self.length, self.step_bc)

        # 上侧
        self.boundary_up = torch.stack(torch.meshgrid(x, y[-1], indexing='ij')).reshape(2, -1).T
        # 下侧
        self.boundary_down = torch.stack(torch.meshgrid(x, y[0], indexing='ij')).reshape(2, -1).T

        x = torch.arange(0, self.length + self.step_bc, self.step_bc)

        # 左侧
        self.boundary_left = torch.stack(torch.meshgrid(x[0], y, indexing='ij')).reshape(2, -1).T
        # 右侧
        self.boundary_right = torch.stack(torch.meshgrid(x[-1], y, indexing='ij')).reshape(2, -1).T
        # 整合
        self.boundary = torch.cat([self.boundary_left, self.boundary_down, self.boundary_up, self.boundary_right])

        # 右侧边界点数值
        self.boundary_stress = torch.arange(self.height, -self.step_bc, -self.step_bc)

        # 数据处理
        self.inside.requires_grad_(True)
        self.inside = self.inside.to(device)
        self.boundary.requires_grad_(True)
        self.boundary = self.boundary.to(device)
        self.boundary_stress.requires_grad_(True)
        self.boundary_stress = self.boundary_stress.to(device)
        # -------------------------- 数据集设置完成 -------------------------- #

        self.iter = 0  # 定义迭代序号，记录调用了多少次loss
        self.loss_history = []  # 用于保存损失值

        # 设置lbfgs优化器
        self.lbfgs = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=5000,
            max_eval=5000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        # 设置adam优化器
        self.adam = torch.optim.Adam(self.parameters(),lr=0.01)

    def forward(self, x):
        x = x.to(device)  # 将输入张量移动到 GPU
        x = self.poly(self.fc1(x))  # 激活函数
        x = self.poly(self.fc2(x))  # 激活函数
        x = self.poly(self.fc3(x))  # 激活函数
        x = self.poly(self.fc4(x))  # 激活函数
        x = self.poly(self.fc5(x))  # 激活函数
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

    # 输入坐标调整
    def normalize_data(self, data):
        data_normalized = data.clone()  # 先克隆数据，避免在原始数据上进行in-place操作
        data_normalized[:, 0] = data_normalized[:, 0] - 10
        data_normalized[:, 1] = data_normalized[:, 1] - 2.5
        return data_normalized

    def calculate_sigma(self, u_pred, v_pred, x_interior):
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

    def calculate_f(self, sigma_xx, sigma_yy, sigma_xy, x_interior):
        fxx_x = torch.autograd.grad(sigma_xx, x_interior, grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0][:, 0]
        fxy_x = torch.autograd.grad(sigma_xy, x_interior, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 0]
        fxy_y = torch.autograd.grad(sigma_xy, x_interior, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0][:, 1]
        fyy_y = torch.autograd.grad(sigma_yy, x_interior, grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0][:, 1]

        return fxx_x, fxy_x, fxy_y, fyy_y

    # 计算损失函数
    def loss_fn(self):
        # 各边数据点个数
        length_point = int(self.length / self.step)
        height_point = int(self.height / self.step)
        inside_point = (length_point - 1) * (height_point - 1)
        length_point = int(self.length / self.step_bc)
        height_point = int(self.height / self.step_bc)
        left_righr_point = (height_point + 1) * 2
        up_down_point = (length_point - 1) * 2

        x_interior = torch.cat((self.inside, self.boundary), dim=0)
        x_interior.requires_grad_(True)
        x_interior = self.normalize_data(x_interior).to(device)  # 确保 x_interior 在 GPU 上

        # 计算损失
        pred = self(x_interior)
        pred.requires_grad_(True)
        pred = pred.to(device)

        u_pred, v_pred = pred[:, 0], pred[:, 1]
        sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = pred[:, 2], pred[:, 3], pred[:, 4]

        sigma_xx, sigma_yy, sigma_xy = self.calculate_sigma(u_pred, v_pred, x_interior)
        fxx_x_pred, fxy_x_pred, fxy_y_pred, fyy_y_pred \
            = self.calculate_f(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, x_interior)
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
        pred_right = pred[right_point:, ]
        sigma_xx_pred_right, sigma_yy_pred_right, sigma_xy_pred_right \
            = pred_right[:, 2], pred_right[:, 3], pred_right[:, 4]

        balence_F_loss = (torch.mean((sigma_xx_pred_right - self.boundary_stress) ** 2) +
                          torch.mean(sigma_yy_pred_right ** 2) +
                          torch.mean(sigma_xy_pred_right ** 2))

        # 上下端应力损失
        pred_up_down = pred[up_point: right_point, ]
        sigma_xx_pred_up_down, sigma_yy_pred_up_down, sigma_xy_pred_up_down \
            = pred_up_down[:, 2], pred_up_down[:, 3], pred_up_down[:, 4]
        balence_up_down_F_loss = (torch.mean((sigma_yy_pred_up_down) ** 2) + torch.mean((sigma_xy_pred_up_down) ** 2))

        # 固定端条件损失
        fixed_pred = pred[inside_point: up_point, ]

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

        return PINN_loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, uv_loss

    def loss_func_adams(self):
        # 将导数清零
        self.adam.zero_grad()
        # 损失函数
        loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, uv_loss = self.loss_fn()

        # loss反向传播，用于给优化器提供梯度信息
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # 梯度裁剪
        # 在控制台上输出消息
        if self.iter % 1000 == 0:
            self.step_train_time = time.time() - self.step_start_time
            print(self.iter,
                  f'Loss: {loss.item():.6f}, '
                  f'balence_without_F_loss: {balence_without_F_loss.item():.6f}, '
                  f'fixed_loss: {fixed_loss.item():.6f}, '
                  f'balence_F_loss: {balence_F_loss.item():.6f}, '
                  f'phy_loss: {phy_loss.item():.6f}, '
                  f'uv_loss: {uv_loss.item():.6f}, '
                  f'Time Elapsed: {self.step_train_time:.2f}s')
            self.step_start_time = time.time()

        self.loss_history.append(loss.item())  # 保存当前损失值
        self.iter = self.iter + 1
        return loss

    def loss_func_lbfgs(self):
        # 检查迭代次数是否超过maxiters
        if self.iter > self.maxiters:
            raise StopIteration("Reached the maximum iteration count for L-BFGS")

        # 将导数清零
        self.lbfgs.zero_grad()

        # 损失函数
        loss, balence_without_F_loss, fixed_loss, balence_F_loss, phy_loss, uv_loss = self.loss_fn()

        # loss反向传播，用于给优化器提供梯度信息
        loss.backward(retain_graph=True)

        if self.iter % 1000 == 0:
            self.step_train_time = time.time() - self.step_start_time
            print(self.iter,
                  f'Loss: {loss.item():.6f}, '
                  f'Time Elapsed: {self.step_train_time:.2f}s')
            self.step_start_time = time.time()

        self.loss_history.append(loss.item())  # 保存当前损失值
        self.iter = self.iter + 1
        return loss

    def plot_results(self):
        x = np.linspace(0, 20, 1000)
        y = np.linspace(0, 5, 1000)
        X, Y = np.meshgrid(x, y)

        xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
        xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True).to(device)
        xy_tensor = self.normalize_data(xy_tensor)
        pred = self(xy_tensor)
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
        plt.plot(self.loss_history[:])
        plt.title('Loss Function Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.yscale('log')  # 可以使用对数坐标
        plt.grid()
        plt.savefig('loss_curve.png')  # 保存图像
        plt.show()  # 显示图像

    def train(self):
        super().train()  # 设置模型为训练模式

        # 首先运行Adam优化器
        print("采用Adam优化器")
        try:
            for epoch in range(self.maxiters):
                if epoch < self.adamsnum + 1:
                    self.adam.step(self.loss_func_adams)
                    if epoch == self.adamsnum:
                        adams_time = time.time() - self.adams_start_time
                        print(f'Adams time: {adams_time:.2f}s')
                else:
                    # L-BFGS优化器
                    self.lbfgs.step(self.loss_func_lbfgs)
        except StopIteration as e:
            print(str(e))
            print("Training stopped due to reaching max iteration.")

        self.plot_results()

start_time = time.time()
# 实例化PINN
pinn = PINN()

# 开始训练
pinn.train()

# 将模型保存到文件
torch.save(pinn, '../Kaiti/5outputModel.pth')
total_time = time.time() - start_time
print(f'Time Elapsed: {total_time:.2f}s')

