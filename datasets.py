from __future__ import annotations

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

def load_state_datas(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray]:

    with h5py.File(file_path, 'r') as f:
        trajectory = f['trajectory'][:]
        masses = f['masses'][:]

    return trajectory, masses

class StateDataset(Dataset):
    def __init__(
        self, file_path: str,
        extra_datas_len: int = 3,
        is_extra_datas: bool = True
    ) -> None:
        trajectory, masses = load_state_datas(file_path)
        self.trajectory = trajectory
        self.masses = masses.reshape(
            trajectory.shape[1], 1
        )

        self.samples_num = trajectory.shape[0]
        self.n_body = trajectory.shape[1]
        self.extra_datas_len = extra_datas_len

        if is_extra_datas:
            self.extra_datas()
        else:
            self.x_len = 7

    def __len__(self) -> int:
        return self.samples_num - 1

    def __getitem__(
            self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.datas[index], self.datas[index + 1]

    def extra_datas(self) -> None:
        l_masses = np.tile(
            self.masses,
            (self.samples_num, 1)
        ).reshape(self.samples_num, self.n_body, 1)

        state_datas = np.concatenate(
            (self.trajectory, l_masses),
            axis=2
        )

        extra_len = 7 + self.extra_datas_len * 2
        extra_datas = np.zeros(
            (self.samples_num,
             self.n_body, self.extra_datas_len *2 + 7)
        )

        for i in range(self.samples_num):
            for j in range(self.n_body):
                r = state_datas[i, j, :3]
                r_norm = np.linalg.norm(r)
                m = state_datas[i, j, 6]
                extra_datas[i, j, 0] = int(r_norm / 10)
                extra_datas[i, j, 1] = int(m / 10)
                extra_datas[i, j, 2] = int(r_norm * m / 20)

                for k in range(3, extra_len):
                    extra_datas[i, j, k] = np.random.uniform(5.7, 6.0)

        datas = np.concatenate(
            (state_datas, extra_datas),
            axis=2
        )

        self.x_len = 14 + self.extra_datas_len * 2
        self.datas = torch.from_numpy(datas).float()

if __name__ == '__main__':
    ts, ms = load_state_datas('Datas/Rk4_test.h5')
    print(ts.shape, ms.shape)

    t = StateDataset('Datas/Rk4_test.h5')
    print(t.samples_num)
    print(t[-1])
