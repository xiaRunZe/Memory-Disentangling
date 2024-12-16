import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from fmridataset import fmri_dataset
import torch.nn.functional as F
from zero_dataset import zero_dataset
import argparse
from basemodel import base_disentangle
from transformers import AdamW, get_linear_schedule_with_warmup



def decode(dataset,gpu = 0,out_dir = '.',k=3,ckpt=0):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device = torch.device(f'cuda:{gpu}')
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
    model = base_disentangle(indim=7604,hidden=2048,outdim=768,window_k=k)
    model.load_state_dict(torch.load(f'out_dir/zerobase-{ckpt:03d}', map_location=torch.device('cpu')))
    model = model.to(device)
    for idx, fdata in enumerate(test_dataloader):
        _, clip_feat, roi,img_idx = fdata
        roi = roi.to(device, dtype=torch.float32)
        clip_feat = clip_feat.to(device, dtype=torch.float32)
        out = model(roi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',type=int,default=4)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--out_dir',type=str,default='./BaseZero')
    args = parser.parse_args()
    out_dir = args.out_dir
    k = args.k
    gpu = args.gpu
    dataset = zero_dataset('test',test_num=300,k = k)
    decode(dataset,gpu,out_dir,k)

