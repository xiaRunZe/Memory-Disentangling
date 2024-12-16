import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pretrain_zerodata import pretrain_dataset
import argparse
from basemodel import base_disentangle,base_cnn
from transformers import AdamW, get_linear_schedule_with_warmup
from scipy.stats import pearsonr


def r2(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def batch_r2(predictions, targets):
    batch_r2 = []
    for i in range(predictions.shape[0]):
        ss_tot = torch.sum((targets[i] - targets[i].mean()) ** 2)
        ss_res = torch.sum((predictions[i] - targets[i]) ** 2)
        if ss_tot.eq(0).any():
            continue
        r2 = 1 - ss_res / ss_tot
        batch_r2.append(r2)
    return sum(batch_r2)/len(batch_r2)
def batch_pcc(predictions, targets):
    batch_pcc = []
    batch_p = []
    for i in range(predictions.shape[0]):
        corr_matrix,p = pearsonr(predictions[i], targets[i])
        batch_pcc.append(corr_matrix)
        batch_p.append(p)
    return sum(batch_pcc)/len(batch_pcc)


def train(args,dataset,lr=1e-4,bs=16,gpu = 0,out_dir = '.',k=3,t=0, subj='subj01',seed=42):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device = torch.device(f'cuda:{gpu}')

    # dim = dim.unsqueeze(0)
    # dim = (1,) + dim
    _, clip_feat, roi, img_idx = dataset[0]
    roi_dim = roi.shape[-1]
    model = base_disentangle(indim=roi_dim,hidden=roi_dim//4,outdim=clip_feat.shape[-1],window_k=k,n_blocks=2)
    model = model.to(device)
    model.train()
    # for param in model.parameters():
    #     param.requires_grad = True
    max_norm = 1
    criterion1 = nn.MSELoss()
    optimizer = AdamW(model.parameters(),lr = lr)
    warm_up_ratio = 0.1

    for epoch in range(epochs):
        print('epoch:',epoch)
        sys.stdout.flush()
        dataset.set_dataset('train')
        train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_ratio * epochs * len(train_dataloader),
            num_training_steps=epochs * len(train_dataloader)
        )
        progress = tqdm(total=len(train_dataloader), desc='train loss')
        for idx,fdata in enumerate(train_dataloader):
            model.zero_grad()
            _,clip_feat,roi,img_idx = fdata
            # print(img_idx.shape)
            # print(fmri3d.shape) #torch.Size([16, 1, 83, 104, 81, k])
            roi = roi.to(device,dtype=torch.float32)
            clip_feat = clip_feat.to(device,dtype = torch.float32)
            # print(roi.shape)
            # print(clip_feat.shape)
            # print(fmri3d)
            out = model(roi[:,0,:])
            # print(out[0].shape)
            losses = []
            for i in range(k):
                loss_i = criterion1(clip_feat[:,i,:],out[i])
                losses.append(loss_i)

            # recon_x, mu, logvar = model(fmri3d)
            loss = losses[0]
            for i in range(1,k) :
                loss = loss + losses[i]

            # print(loss)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss_base": loss.item()})
            progress.update()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"zerobase-{epoch:03d}.pt"),
            )
            dataset.set_dataset('test')
            test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
            progress.close()
            progress = tqdm(total=len(test_dataloader), desc='dev loss')
            model.eval()
            # mean_r20 = []
            # mean_r21 = []
            # mean_r22 = []
            mean_r = [[] for i in range(k)]
            mean_pcc = [[] for i in range(k)]
            base_clip = []
            for idx, fdata in enumerate(test_dataloader):
                _, clip_feat, roi,img_idx = fdata
                # print(img_idx.shape)
                # print(fmri3d.shape) #torch.Size([16, 1, 83, 104, 81, k])
                roi = roi.to(device, dtype=torch.float32)
                clip_feat = clip_feat.to(device, dtype=torch.float32)
                # print(roi.shape)
                # print(clip_feat.shape)
                # print(fmri3d)
                outs = model(roi[:,0,:])
                loss0 = criterion1(clip_feat[:, 0, :], outs[0])
                loss1 = criterion1(clip_feat[:, 1, :], outs[1])
                temp = []
                for out in outs:
                    temp.append(out.cpu().detach().numpy())
                base_clip.append(temp)
                for i in range(k):
                    mean_r[i].append(batch_r2(outs[i],clip_feat[:, i, :]))
                    mean_pcc[i].append(batch_pcc(outs[i].cpu().detach().numpy(), clip_feat[:, i, :].cpu().detach().numpy()))

                progress.set_postfix({"loss1": loss1.item(), "loss0": loss0.item()})
                progress.update()
            for i in range(k):
                r = sum(mean_r[i]) / len(mean_r[i])
                pcc = sum(mean_pcc[i]) / len(mean_pcc[i])
                print(f'r{i}:', r,f'pcc{i}:',pcc)
            base_clip = np.array(base_clip)
            base_clip = base_clip[:, :, 0, :]
            print(base_clip.shape)
            if base_clip.shape[-1] == 768:
                np.save(f'Decoded_clip_{subj}/clip_base_0_ckpt{epoch}_test{t}_{seed}.npy', base_clip)
            elif base_clip.shape[-1] == 77 * 768:
                np.save(f'Decoded_clip_{subj}/c_base_0_ckpt{epoch}_test{t}_{seed}.npy', base_clip)
            model.train()
            progress.close()

def train_cnn(args,dataset,lr=1e-4,bs=16,gpu = 0,out_dir = '.',k=3):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device = torch.device(f'cuda:{gpu}')

    # dim = dim.unsqueeze(0)
    # dim = (1,) + dim
    _, clip_feat, roi, img_idx = dataset[0]
    roi_dim = roi.shape[-1]
    model = base_cnn(in_dim=roi_dim,hidden=roi_dim//4,out_dime=roi_dim//8,out_dimd=clip_feat.shape[-1],n_blocks=1,window_k=k)
    model = model.to(device)
    model.train()
    max_norm = 1
    criterion1 = nn.MSELoss()
    # criterion = nn.functional.cosine_similarity()
    optimizer = AdamW(model.parameters(),lr = lr)
    warm_up_ratio = 0.1

    for epoch in range(epochs):
        print('epoch:',epoch)
        sys.stdout.flush()
        dataset.set_dataset('train')
        train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_ratio * epochs * len(train_dataloader),
            num_training_steps=epochs * len(train_dataloader)
        )
        progress = tqdm(total=len(train_dataloader), desc='train loss')
        for idx,fdata in enumerate(train_dataloader):
            model.zero_grad()
            _,clip_feat,roi,img_idx = fdata
            roi = roi.to(device,dtype=torch.float32)
            clip_feat = clip_feat.to(device,dtype = torch.float32)
            out = model(roi[:,0,:])
            # print(out[0].shape)
            losses = []
            for i in range(k):
                loss_i = criterion1(clip_feat[:,i,:],out[i])
                cosine_loss_i = 1 - torch.nn.functional.cosine_similarity(clip_feat[:, i, :], out[i], dim=1).mean()
                losses.append(loss_i+cosine_loss_i)
            loss = losses[0]
            for i in range(1,k) :
                loss = loss + losses[i]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss_base": loss.item()})
            progress.update()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"zerobase-{epoch:03d}.pt"),
            )
        dataset.set_dataset('test')
        test_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=True)
        progress.close()
        progress = tqdm(total=len(test_dataloader), desc='dev loss')
        model.eval()
        # mean_r20 = []
        # mean_r21 = []
        # mean_r22 = []
        mean_r = [[] for i in range(k)]
        mean_pcc = [[] for i in range(k)]
        for idx, fdata in enumerate(test_dataloader):
            _, clip_feat, roi,img_idx = fdata
            roi = roi.to(device, dtype=torch.float32)
            clip_feat = clip_feat.to(device, dtype=torch.float32)
            out = model(roi[:,0,:])
            for i in range(k):
                mean_r[i].append(batch_r2(out[i],clip_feat[:, i, :]))
                mean_pcc[i].append(batch_pcc(out[i].cpu().detach().numpy(), clip_feat[:, i, :].cpu().detach().numpy()))
            loss0 = criterion1(clip_feat[:, 0, :], out[0])
            loss1 = criterion1(clip_feat[:, 1, :], out[1])
            # loss2 = criterion1(clip_feat[:, 2, :], out[2])
            # print(loss)
            progress.set_postfix({"loss1": loss1.item(), "loss0": loss0.item()})
            progress.update()
        for i in range(k):
            r = sum(mean_r[i]) / len(mean_r[i])
            pcc = sum(mean_pcc[i]) / len(mean_pcc[i])
            print(f'r{i}:', r,f'pcc{i}:',pcc)
        model.train()
        progress.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--bs',type=int,default=64)
    parser.add_argument('--k',type=int,default=4)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--save_every',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--subj',type=str,default='subj01')
    parser.add_argument('--out_dir',type=str,default='./ZeroBase')

    args = parser.parse_args()
    out_dir = args.out_dir
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    k = args.k
    gpu = args.gpu
    subjs = ['subj07','subj02','subj05','subj01',]
    seeds = [ 789,159,42, 99,265]
    test = 1
    os.makedirs(args.out_dir,exist_ok=True)
    subj = args.subj
    test_num = 500
    for subj in subjs:
        for seed in seeds:
            dataset = pretrain_dataset('train', subj=subj, test_num=test_num, k=k, seed=seed)
            for t in range(test):
                train(args, dataset, lr, bs, gpu, out_dir, k, t,subj,seed)
    # dataset = pretrain_dataset('train',subj=subj,test_num=test_num,seed=args.seed,k = k)
    # dataset.set_dataset('train')
    # for t in range(test):
    #     train(args,dataset,lr,bs,gpu,out_dir,k,t)
    # train_cnn(args,dataset,lr,bs,gpu,out_dir,k)
