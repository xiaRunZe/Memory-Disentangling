import torch.nn as nn
import torch
import numpy

class MLP(torch.nn.Module):
    def __init__(self, in_dim=7604, h=4096,out_dim=6400, n_blocks=1):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                # nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ) for _ in range(n_blocks)
        ])
        self.linear1 = nn.Linear(h, h)
        self.final_mlp = nn.Sequential(
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = self.linear1(x)
        x = x.reshape(len(x), -1)
        x = self.final_mlp(x)
        return x

class base_disentangle(nn.Module):  #used base model in paper
    def __init__(self,indim=7604,hidden=2048,outdim=768,window_k=3,n_blocks = 2):
        super().__init__()
        self.mlp_trans = nn.ModuleList([
            nn.Sequential(
                MLP(indim,hidden,outdim,n_blocks=n_blocks),
            ) for _ in range(window_k)
            ])
        self.window = window_k
    def forward(self, x):
        semantics = []
        for i in range(self.window):
            now_semantic = self.mlp_trans[i](x)
            semantics.append(now_semantic)
        return semantics

class base_cnn(nn.Module):
    def __init__(self, in_dim,hidden, out_dime,out_dimd,n_blocks,window_k):
        super(base_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * (in_dim // 4), out_dime)  # Adjust based on the new dimensions
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.window_k = window_k

        self.mlp_trans = nn.ModuleList([
            nn.Sequential(
                MLP(out_dime, hidden, out_dimd, n_blocks=n_blocks),
            ) for _ in range(window_k)
        ])
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        semantics = []
        for i in range(self.window_k):
            now_semantic = self.mlp_trans[i](x)
            semantics.append(now_semantic)
        return semantics

