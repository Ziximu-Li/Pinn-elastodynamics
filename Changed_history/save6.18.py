import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim import LBFGS

# 定义神经网络结构
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 40)  # 输入层到隐藏层
        self.fc2 = nn.Linear(40, 40)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(40, 40)  # 隐藏层到隐藏层
        self.fc4 = nn.Linear(40, 5)  # 隐藏层到输出层，输出5个分量：两个位移分量和三个应力分量

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 激活函数tanh
        x = torch.tanh(self.fc2(x))  # 激活函数tanh
        x = torch.tanh(self.fc3(x))  # 激活函数tanh
        x = self.fc4(x)  # 输出层
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 使用Xavier初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始化偏置为0

# 生成训练数据
def generate_BC_training_data(length, height, step):

    # 固定端边界条件 (x = 0)    length = 20, height = 5, step = 200
    x_boundary_fixed = torch.zeros((step, 2))  # 边界坐标点
    x_boundary_fixed[:, 1] = torch.linspace(0, height, step)  # 生成 x=0 边界的坐标  --> [0, y]T
    x_boundary_fixed.requires_grad_(True)
    y_boundary_fixed = torch.zeros((step, 2))  # 设置固定端的位移: y_boundary_fixed --> U and V (x = 0) = 0

    # 其他边界条件 (仅应力)
    x_boundary_stress = torch.cat([
        torch.cat([torch.linspace(0, length, step)[:, None], torch.zeros((step, 1))], dim=1),  # 下边界
        torch.cat([torch.linspace(0, length, step)[:, None], height * torch.ones((step, 1))], dim=1),  # 上边界
        torch.cat([length * torch.ones((step, 1)), torch.linspace(0, height, step)[:, None]], dim=1)  # 右边界
    ]) # 定义上下右侧边界：x_boundary_stress -->[sigma_yy(y=0) + sigma_yy(y=5) + sigma_xx(x=20)] --> [200*3, 2]T
    x_boundary_stress.requires_grad_(True)
    y_boundary_stress = torch.zeros((3 * step, 1))  # 应力边界条件的大小  --> [200*3, 1]T

    # y_boundary_stress[:step, 1] = 0.0  # 下边界 \(\sigma_{yy} = 0\)
    # y_boundary_stress[step:2 * step, 1] = 0.0  # 上边界 \(\sigma_{yy} = 0\)
    # y_boundary_stress[2 * step:, 0] = 0.0  # 右边界 \(\sigma_{xx} = 0\)

    force_magnitude = 5.0
    angle = 0.0  # 力的方向
    force_x = force_magnitude
    # force_y = -force_magnitude * np.sin(np.radians(angle))  # 向下为负方向
    # check = int((2 + 1 / 2) * step)
    y_boundary_stress[int((2 + 1 / 2) * step), 0] = force_x  # 右侧中点 F = 5N
    y_boundary_stress.requires_grad_(True)

    # 右侧边界中点施加力，仅约束\(\sigma_{xx}\)和\(\sigma_{yy}\)
    # x_boundary_force = torch.tensor([[length, height / 2]], requires_grad=True)  # 右侧边界中点
    # y_boundary_force = torch.tensor([[force_x, force_y]], requires_grad=True)  # 仅约束\(\sigma_{xx}\)和\(\sigma_{yy}\)
    # return x_boundary_fixed, y_boundary_fixed, x_boundary_stress, y_boundary_stress, x_boundary_force, y_boundary_force

    return x_boundary_fixed, y_boundary_fixed, x_boundary_stress, y_boundary_stress

# 定义损失函数，包含PDE损失和边界条件损失
def loss_fn(model, x_interior, x_boundary_fixed, y_boundary_fixed, x_boundary_stress, y_boundary_stress):
    # 计算PDE损失
    pred = model(x_interior)
    u_pred, v_pred = pred[:, 0], pred[:, 1]
    sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = pred[:, 2], pred[:, 3], pred[:, 4]

    # 几何方程
    u_x = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(v_pred, x_interior, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v_pred, x_interior, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    epsilon_xx = u_x
    epsilon_yy = v_y
    epsilon_xy = 0.5 * (u_y + v_x)
    gamma_xy = u_y + v_x

    # 本构方程 (线弹性材料，杨氏模量 E 和泊松比 nu)
    E = 2.1e5  # 杨氏模量
    mu = 0.3  # 泊松比

    C11 = E / (1 - mu ** 2)
    C12 = mu * E / (1 - mu ** 2)
    C66 = E / (2 * (1 + mu))

    sigma_xx = C11 * epsilon_xx + C12 * epsilon_yy
    sigma_yy = C12 * epsilon_xx + C11 * epsilon_yy
    tau_xy = C66 * gamma_xy

    constitutive_loss = torch.mean((sigma_xx_pred - sigma_xx) ** 2) + \
                        torch.mean((sigma_yy_pred - sigma_yy) ** 2) + \
                        torch.mean((sigma_xy_pred - tau_xy) ** 2)

    # 平衡方程
    f_x = torch.autograd.grad(sigma_xx_pred, x_interior, grad_outputs=torch.ones_like(sigma_xx_pred), create_graph=True)[0][:, 0] + \
          torch.autograd.grad(sigma_xy_pred, x_interior, grad_outputs=torch.ones_like(sigma_xy_pred), create_graph=True)[0][:, 1]
    f_y = torch.autograd.grad(sigma_xy_pred, x_interior, grad_outputs=torch.ones_like(sigma_xy_pred), create_graph=True)[0][:, 0] + \
          torch.autograd.grad(sigma_yy_pred, x_interior, grad_outputs=torch.ones_like(sigma_yy_pred), create_graph=True)[0][:, 1]

    # PDE损失
    pde_loss = torch.mean(f_x ** 2) + torch.mean(f_y ** 2) + constitutive_loss

    # 固定端边界条件损失 (仅位移)
    bc_pred_fixed = model(x_boundary_fixed)
    bc_loss_fixed = torch.mean((bc_pred_fixed[:, :2] - y_boundary_fixed) ** 2)

    # 其他边界条件损失 (仅应力)
    bc_pred_stress = model(x_boundary_stress)
    # y_boundary_stress[:step, 1] = 0.0  # 下边界 \(\sigma_{yy} = 0\)
    # y_boundary_stress[step:2 * step, 1] = 0.0  # 上边界 \(\sigma_{yy} = 0\)
    # y_boundary_stress[2 * step:, 0] = 0.0  # 右边界 \(\sigma_{xx} = 0\)

    bc_pred_stress_use = torch.cat((
        bc_pred_stress[:step, 3],
        bc_pred_stress[step:2 * step, 3],
        bc_pred_stress[2 * step:, 2]), dim = 0)
    bc_loss_stress = torch.mean((bc_pred_stress_use - y_boundary_stress) ** 2)

    lambda_fixed = 0.1  # 固定端边界条件权重
    lambda_stress = 0.1  # 应力边界条件权重

    return pde_loss + lambda_fixed * bc_loss_fixed + lambda_stress * bc_loss_stress

# 主训练过程
def train(maxiters, n, steps_per_iteration, step):
    model = PINN()  # 初始化PINN模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    loss_history = []

    steps_count = maxiters // n  # 每次加点训练的次数
    num_print_epoch = steps_count // 4

    # 悬臂梁尺寸
    length = 20.0
    height = 5.0

    # 边界点取点数
    x_boundary_fixed, y_boundary_fixed, x_boundary_stress, y_boundary_stress = generate_BC_training_data(length, height, step)

    # 初始化空的内部点
    x_interior_total = torch.empty(0, 2)

    for i in range(n):
        step_train_start_time = time.time()

        x_interior = torch.rand((steps_per_iteration, 2))  # 随机生成内部点
        x_interior[:, 0] *= length
        x_interior[:, 1] *= height
        x_interior.requires_grad_(True)

        x_interior_total = torch.cat((x_interior_total, x_interior), dim=0)  # 将新生成的点加入总的内部点集合
        x_interior_total.requires_grad_(True)

        if i < n - 1  :
            for epoch in range(steps_count):
                optimizer.zero_grad()
                loss = loss_fn(model, x_interior_total, x_boundary_fixed, y_boundary_fixed, x_boundary_stress, y_boundary_stress)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                loss_history.append(loss.item())

                if epoch % num_print_epoch == 0:
                    print(f'Iteration {i + 1}/{n}, Epoch {epoch}, Loss: {loss.item():.5f}')
        else:
            optimizer = LBFGS(model.parameters(), lr=0.1, max_iter=500, history_size=50, line_search_fn='strong_wolfe')  # 最后一轮使用LBFGS优化器

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(model, x_interior_total, x_boundary_fixed, y_boundary_fixed, x_boundary_stress, y_boundary_stress)
                loss.backward()
                return loss

            for epoch in range(steps_count):
                loss = optimizer.step(closure)  # LBFGS优化

                loss_history.append(loss.item())

                if epoch % num_print_epoch == 0:
                    print(f'Iteration {i + 1}/{n}, Epoch {epoch}, Loss: {loss.item():.5f}')

        step_train_time = time.time() - step_train_start_time
        print(f'Step {i + 1}/{n}, Time Elapsed: {step_train_time:.2f}s')

    return model, loss_history

# 绘制应力和位移云图
def plot_results(model, loss_history):
    x = np.linspace(0, 20, 1000)
    y = np.linspace(0, 5, 1000)
    X, Y = np.meshgrid(x, y)

    xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    xy_tensor = torch.tensor(xy, dtype=torch.float32, requires_grad=True)
    pred = model(xy_tensor)
    U = pred[:, 0].reshape(1000, 1000).detach().numpy()
    V = pred[:, 1].reshape(1000, 1000).detach().numpy()
    sigma_xx = pred[:, 2].reshape(1000, 1000).detach().numpy()
    sigma_yy = pred[:, 3].reshape(1000, 1000).detach().numpy()
    sigma_xy = pred[:, 4].reshape(1000, 1000).detach().numpy()

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

    # 绘制损失函数曲线，从第1000次训练后开始
    plt.figure()
    plt.plot(loss_history[1000:])
    plt.title('Loss Function (After 1000 Iterations)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    num_phi_train = 2000  # 总加点个数
    n = 4  # 加点轮次
    maxiters = 4000  # 总共训练的次数
    steps_per_iteration = num_phi_train // n  # 单次加点个数
    step = 200  # 定义边界加点个数

    start_time = time.time()
    model, loss_history = train(maxiters, n, steps_per_iteration, step)  # 传递step参数
    elapsed_time = time.time() - start_time
    print(f'Time Elapsed: {elapsed_time:.2f}s')
    plot_results(model, loss_history)

