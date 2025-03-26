from __future__ import annotations

import torch
import torch.nn as nn

class AFTsimple(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 初始化投影层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        输入: x - (batch_size, seq_len, d_model)
        输出: y - (batch_size, seq_len, d_model)
        """

        Q = torch.sigmoid(self.wq(x))  # 门控投影 [B,T,d]
        K = torch.exp(self.wk(x))      # 指数变换 [B,T,d]
        V = self.wv(x)                 # 值投影 [B,T,d]

        # 计算全局上下文
        sum_kv = torch.sum(K * V, dim=1, keepdim=True)  # 分子项 [B,1,d]
        sum_k = torch.sum(K, dim=1, keepdim=True)       # 分母项 [B,1,d]

        # 稳定化处理（防止除零）
        epsilon = 1e-5
        weighted = sum_kv / (sum_k + epsilon)  # 全局聚合 [B,1,d]

        # 门控输出
        y = Q * weighted + x  # 广播乘法 [B,T,d]
        return y


class AsterPredNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        x_len: int,
        seq_len: int,
        num_layers: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
            nn.Linear(x_len, d_model),
            nn.Sigmoid(),
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid(),
            nn.Linear(d_model * 2, d_model),
        )

        self.attn_layer = AFTsimple(d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid(),
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(0.5),
            nn.Linear(d_model, d_model // 2),
            nn.Linear(d_model // 2, x_len),
        )

    def forward(self, x: torch.Tensor):
        """
        输入: x - (batch_size, seq_len, x_len)
        输出: y - (batch_size, seq_len, x_len)
        """

        # 编码器
        x = self.encoder(x)  # [B,T,d]
        # 注意力层
        for _ in range(self.num_layers):
            x = self.attn_layer(x)  # [B,T,d]
        # 解码器
        y = self.decoder(x)  # [B,T,x_len]

        return y

if __name__ == '__main__':
    from datasets import StateDataset

    # 定义数据集
    dataset = StateDataset("./Datas/Rk4_test.h5")
    # 定义模型
    model = AsterPredNet(
        d_model=128, x_len=dataset.x_len,
        seq_len=dataset.n_body, num_layers=2
    )

    x1, y1 = dataset[0]
    print(x1.shape, y1.shape)
    y_pred = model(x1)
    print(y_pred.shape)

