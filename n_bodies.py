from __future__ import annotations

import numpy as np
from numba import njit, prange

import h5py
import time

# state:
# [[x1, y1, z1, vx1, vy1, vz1],
#  [x2, y2, z2, vx2, vy2, vz2],
#  [x3, y3, z3, vx3, vy3, vz3],
#  [x4, y4, z4, vx4, vy4, vz4],
#  ...]

@njit(parallel=True)
def compute_derivative(
    state: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    eps: float = 1e-6,
):
    """
    计算三体系统的状态导数，使用Numba加速和多线程并行化。
    state: 三体系统的状态，shape=(n_bodies, 6)
    masses: 三体系统的质量，shape=(n_bodies,)
    G: 万有引力常数(默认值1.0)
    eps: 微小值，防止除零错误(默认值1e-6)
    """

    n_bodies = state.shape[0]
    deriv = np.zeros_like(state)
    for i in prange(n_bodies):  # 并行循环处理每个天体
        x_i, y_i, z_i = state[i, :3]
        ax, ay, az = 0.0, 0.0, 0.0
        
        # 计算其他天体对当前天体的引力作用
        for j in range(n_bodies):
            # 跳过自身
            if j == i: continue

            x_j, y_j, z_j = state[j, :3]
            dx = x_j - x_i
            dy = y_j - y_i
            dz = z_j - z_i
            r_sq = dx**2 + dy**2 + dz**2 + eps
            r_cubed = r_sq ** 1.5  # (r^2)^(3/2) = r^3
            factor = G * masses[j] / r_cubed
            ax += factor * dx
            ay += factor * dy
            az += factor * dz

        # 更新导数：位置导数为速度，速度导数为加速度
        deriv[i] = [
            state[i, 3],
            state[i, 4],
            state[i, 5],
            ax, ay, az
        ]

    return deriv

@njit
def rk4_step(
    state: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    dt: float = 0.001,
):
    """
    使用四阶Runge-Kutta方法进行单步积分。
    state: 三体系统的状态，shape=(n_bodies, 6)
    masses: 三体系统的质量，shape=(n_bodies,)
    G: 万有引力常数(默认值1.0)
    dt: 时间步长(默认值0.001)
    """
    k1 = compute_derivative(state, masses, G)
    k2 = compute_derivative(state + 0.5*dt*k1, masses, G)
    k3 = compute_derivative(state + 0.5*dt*k2, masses, G)
    k4 = compute_derivative(state + dt*k3, masses, G)

    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

@njit
def simulate(
    initial_state: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    dt: float = 0.001,
    steps: int = 1000,
    gap: int = 100,
):
    """
    执行完整模拟并返回轨迹数据。
    initial_state: 三体系统的初始状态，shape=(n_bodies, 6)
    masses: 三体系统的质量，shape=(n_bodies,)
    G: 万有引力常数(默认值1.0)
    dt: 时间步长(默认值0.001)
    steps: 步数(默认值1000)
    gap: 记录步数(默认值100)
    """

    trajectory = np.zeros((steps // gap, initial_state.shape[0], 6))
    current_state = initial_state.copy()
    for step in range(steps):
        current_state = rk4_step(current_state, masses, G, dt)
        if (step + 1) % gap == 0:
            trajectory[step//gap] = current_state

    return trajectory


def save_trajectory(
    trajectory: np.ndarray,
    masses: np.ndarray,
    energies: np.ndarray,
    filename: str,
):
    """
    将轨迹数据保存到HDF5文件。
    trajectory: 轨迹数据，shape=(steps, n_bodies, 6)
    filename: 文件名
    """

    with h5py.File(filename, 'w') as f:
        f.create_dataset('trajectory', data=trajectory)
        f.create_dataset('masses', data=masses)
        f.create_dataset('energies', data=energies)

@njit
def total_energy(state, masses, G):
    n = len(masses)
    ke = 0.0  # 动能
    pe = 0.0  # 势能
    for i in range(n):
        vx, vy, vz = state[i, 3:6]
        ke += 0.5 * masses[i] * (vx**2 + vy**2 + vz**2)
        for j in range(i+1, n):
            dx = state[i, 0] - state[j, 0]
            dy = state[i, 1] - state[j, 1]
            dz = state[i, 2] - state[j, 2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            pe -= G * masses[i] * masses[j] / r

    return ke + pe

if __name__ == '__main__':
    # 定义初始状态
    initial_state = np.array([
        [0.0, 0.0, 0.0,
         0.1, -1.2, 0.01],
        [0.9, 0.4, 0.6,
         8.3, 8.5, 3.1],
        [1.1, 1.3, -1.7,
         -12.0, 12.0, -13.1],
        [2.8, 2.5, -3.2,
         -12.0, 14.1, 1.2],
        [-4.6, 5.1, 6.0,
         -18.9, 10.6, -1.1],
        [-7.1, 9.8, -7.7,
         -19.6, -11.7, 11.9],
        [5.5, 3.7, 4.5,
         -14.1, -9.4, 12.4],
        [6.9, 7.9, 6.2,
         -11.0, 13.8, -0.1],
    ])
    n_bodies = initial_state.shape[0]

    # 定义时间步长和步数
    mode = "Train_Data"
    #mode = "Test_Data"
    if mode == "Train_Data":
        dt = 1e-6
        steps = 1000000
        gap = 100
        masses = np.array([
            120.0, 31.9, 9.8, 10.8,
            2.14, 2.03, 2.01, 2.12,
        ])

    else:
        dt = 1e-6
        steps = 10000
        gap = 100
        initial_state = np.array([
            [0.0, 0.0, 0.0,
             -39.5, -12.1, 15.5],
            [0.8, 0.9, 0.7,
             0.2, 0.2, 0.1],
            [0.3, -1.0, 0.8,
             -22.8, -33.9, 35.9],
            [2.8, 1.7, 1.8,
             -23.7, 38.9, 38.8],
            [-6.7, -6.1, 3.0,
             -13.9, 39.3, 25.3],
            [-4.5, 4.5, -5.0,
             -18.0, -27.9, 26.1],
            [3.1, 5.6, 6.0,
             -27.3, -26.9, 25.8],
            [7.5, 1.3, 4.0,
             -26.8, 26.4, 24.5],
        ])
        masses = np.array([
            29.8, 99.8, 9.2, 10.4,
            3.18, 2.18, 2.21, 2.12,
        ])

    # 运行模拟
    print(f"Running simulation with {steps} steps and {dt} time step...")
    start_time = time.time()
    trajectory = simulate(
        initial_state, masses,
        dt=dt, steps=steps,
        gap=gap,
    )
    # 打印最终位置
    for i in range(n_bodies):
        print(f"Final position of body {i}: x={trajectory[-1, i, 0]:.4f}, y={trajectory[-1, i, 1]:.4f}, z={trajectory[-1, i, 2]:.4f}")

    # 计算总能量
    first_energy = total_energy(initial_state, masses, G=1.0)
    final_energy = total_energy(trajectory[-1], masses, G=1.0)
    print(f"Final energy: {final_energy:.4f} (initial: {first_energy:.4f})")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    energies = np.array([first_energy, final_energy])

    # 保存轨迹数据
    if mode == "Train_Data":
        save_trajectory(trajectory, masses, energies, './Datas/Rk4_train.h5')
        print(f"Trajectory saved to ./Datas/Rk4_train.h5")
    else:
        save_trajectory(trajectory, masses, energies, './Datas/Rk4_test.h5')
        print(f"Trajectory saved to ./Datas/Rk4_test.h5")
