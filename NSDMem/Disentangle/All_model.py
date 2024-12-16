from Disentan_Encoder import Disentan_CNN,Disentan_MLP
from basemodel import MLP
import torch.nn as nn


class disentangle(nn.Module):
    def __init__(self,in_dim=2048,h=1024,out_dim=768,window_k=4):
        super().__init__()
        self.mlp_trans = nn.ModuleList([
            nn.Sequential(
                MLP(in_dim, h, out_dim, n_blocks=1),
            ) for _ in range(window_k)
        ])
        self.window_k = window_k
        self.weight = 0.1
    def forward(self, x_a,x_b):
        semantics = []
        now_semantic = self.mlp_trans[0](x_a)
        semantics.append(now_semantic)
        for i in range(1,self.window_k):
            now_semantic = self.mlp_trans[i](x_b)
            semantics.append(now_semantic)
        return semantics

class all_model(nn.Module):
    def __init__(self,model_type='cnn',window_k=3,in_dim=7604,h=4096,out_dime = 2048,out_dimd=768,n=1):
        super().__init__()
        if model_type == 'mlp':
            self.encoder = Disentan_MLP(in_dim,h,out_dime,n)
        else:
            self.encoder = Disentan_CNN(in_dim,out_dime)
        self.decoder = disentangle(out_dime,h,out_dimd,window_k)

    def forward(self,x):
        x_a,x_b = self.encoder(x)
        out = self.decoder(x_a,x_b)
        return out
