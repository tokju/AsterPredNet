import h5py
import numpy as np
from numba import njit
from matplotlib import pyplot as plt

def load_state_datas(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray]:

    with h5py.File(file_path, 'r') as f:
        trajectory = f['trajectory'][:]
        masses = f['masses'][:]

    return trajectory, masses

@njit
def total_energy(state, masses, G = 1.0):
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
    train_file_path = './Datas/Rk4_train.h5'
    test_file_path = './Datas/Rk4_test.h5'
    train_datas = load_state_datas(train_file_path)
    test_datas = load_state_datas(test_file_path)
    train_trajectories = train_datas[0]
    test_trajectories = test_datas[0]
    train_masses = train_datas[1]
    test_masses = test_datas[1]
    print(train_trajectories.shape)
    print(test_trajectories.shape)
    print(train_masses.shape)
    print(test_masses.shape)

    train_energies = [
        total_energy(train_trajectories[i], train_masses)
        for i in range(0, len(train_trajectories), 500)
    ]
    test_energies = [
        total_energy(test_trajectories[i], test_masses)
        for i in range(0, len(test_trajectories), 5)
    ]
    print(len(train_energies))
    print(len(test_energies))

    # mode = 'train'
    mode = 'test'
    if mode == 'train':
        plt.plot(
            np.arange(len(train_energies)),
            train_energies,
            label='train',
            marker='o',
            color='blue'
        )
        plt.title('RK4 method train energy')
        plt.xlabel('step')
        plt.ylabel('energy')
        plt.grid()
        plt.xticks(
            np.arange(0, 20, 1),
        )
        plt.savefig('./logs/Rk4_train_energy.png')
    else:
        plt.plot(
            np.arange(len(test_energies)),
            test_energies,
            label='test',
            marker='o',
            color='red'
        )
        plt.title('RK4 method test energy')
        plt.xlabel('step')
        plt.ylabel('energy')
        plt.grid()
        plt.xticks(
            np.arange(0, 20, 1)
        )
        plt.savefig('./logs/Rk4_test_energy.png')
