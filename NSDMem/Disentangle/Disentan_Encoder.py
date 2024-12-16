import torch.nn as nn
import torch



class CNNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * (in_dim // 4), out_dim)  # Adjust based on the new dimensions
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
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

class Disentan_CNN(torch.nn.Module):
    def __init__(self,in_dim=7604,out_dim=2048):
        super().__init__()
        self.now_CNN = CNNEncoder(in_dim=in_dim,out_dim=out_dim)
        self.before_CNN = CNNEncoder(in_dim=in_dim,out_dim=out_dim)
    def forward(self,x):
        now_embedding = self.now_CNN(x)
        before_embedding = self.before_CNN(x)
        return now_embedding,before_embedding

class Disentan_MLP(torch.nn.Module):
    def __init__(self,in_dim=7604,h = 4096,out_dim=2048,n=1):
        super().__init__()
        self.now_MLP = MLP(in_dim=in_dim,h=h,out_dim=out_dim,n_blocks=n)
        self.before_MLP = MLP(in_dim=in_dim,h=h,out_dim=out_dim,n_blocks=n)
    def forward(self,x):
        now_embedding = self.now_MLP(x)
        before_embedding = self.before_MLP(x)
        return now_embedding,before_embedding

