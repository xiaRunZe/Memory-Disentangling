import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pretrain_zerodata import pretrain_dataset
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
from All_model import all_model
from scipy.stats import pearsonr
from pytorchtools import EarlyStopping

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
    return sum(batch_r2) / len(batch_r2)





class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, before_out1, now_out1, now_out2, before_out2):
        # Normalize the embeddings to unit vectors
        before_out1 = F.normalize(before_out1, dim=1)
        now_out1 = F.normalize(now_out1, dim=1)
        now_out2 = F.normalize(now_out2, dim=1)
        before_out2 = F.normalize(before_out2, dim=1)

        # Calculate positive pair similarity (before_out1, now_out2)
        pos_sim = torch.exp(F.cosine_similarity(before_out1, now_out2) / self.temperature)

        # Calculate negative pair similarities
        neg_sim1 = torch.exp(F.cosine_similarity(now_out1, before_out1) / self.temperature)
        neg_sim2 = torch.exp(F.cosine_similarity(now_out2, before_out2) / self.temperature)
        neg_sim3 = torch.exp(F.cosine_similarity(now_out1, now_out2) / self.temperature)
        neg_sim4 = torch.exp(F.cosine_similarity(before_out1, before_out2) / self.temperature)
        neg_sim5 = torch.exp(F.cosine_similarity(now_out1, before_out2) / self.temperature)

        # Concatenate negative similarities
        neg_sims = torch.cat([neg_sim1, neg_sim2, neg_sim3, neg_sim4, neg_sim5], dim=0)

        # Compute the InfoNCE loss
        loss = -torch.log(pos_sim / torch.sum(neg_sims))  # (pos_sim + torch.sum(neg_sims)))

        return torch.mean(loss)


def batch_pcc(predictions, targets):
    batch_pcc = []
    batch_p = []
    for i in range(predictions.shape[0]):
        corr_matrix, p = pearsonr(predictions[i], targets[i])
        batch_pcc.append(corr_matrix)
        batch_p.append(p)
    return sum(batch_pcc) / len(batch_pcc)



def train_cons(args, dataset, lr=1e-4, bs=16, gpu=0, out_dir='.', k=3, w=0.01, t=0, seed=0,
               subj='subj01'):  # paper used

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device = torch.device(f'cuda:{gpu}')
    _, clip_feat, roi, img_idx = dataset[0]
    roi_dim = roi.shape[-1]
    model = all_model(model_type='mlp', in_dim=roi_dim, h=roi_dim // 4, out_dime=roi_dim // 4,
                      out_dimd=clip_feat.shape[-1], n=2
                      , window_k=k)
    model = model.to(device)
    model.train()
    criterion1 = nn.MSELoss()
    tem = 0.5
    loss_fn = InfoNCELoss(temperature=tem)
    optimizer = AdamW(model.parameters(), lr=lr)
    warm_up_ratio = 0.1
    weight = w
    patience = 80
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    dataset.set_dataset('train')
    train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=1, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_ratio * epochs * len(train_dataloader),
        num_training_steps=epochs * len(train_dataloader))
    for epoch in range(0,epochs):
        print('epoch:', epoch)
        sys.stdout.flush()
        dataset.set_dataset('train')
        progress = tqdm(total=len(train_dataloader), desc='train loss')
        for idx, fdata in enumerate(train_dataloader):
            model.zero_grad()
            _, clip_feat, roi, img_idx = fdata
            roi = roi.to(device, dtype=torch.float32)
            clip_feat = clip_feat.to(device, dtype=torch.float32)
            out = model(roi[:, 0, :])
            now_out1, before_out1 = model.encoder(roi[:, 0, :])
            now_out2, before_out2 = model.encoder(roi[:, 1, :])
            # print(out[0].shape)
            losses = []
            for i in range(k):
                loss_i = criterion1(clip_feat[:, i, :], out[i])
                losses.append(loss_i)
            loss = losses[0]
            for i in range(1, k):
                loss = loss + losses[i]
            loss2 = loss_fn(before_out1, now_out1, before_out2, now_out2)
            loss = weight * loss2 + loss
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        if epoch == 0 or (epoch+1) % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"{subj}_{w}_{seed}_cons_all-{epoch:03d}.pt"),
            )
        dataset.set_dataset('test')
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        progress.close()
        progress = tqdm(total=len(test_dataloader), desc='dev loss')
        model.eval()
        mean_r = [[] for i in range(k)]
        mean_pcc = [[] for i in range(k)]
        base_clip = []
        for idx, fdata in enumerate(test_dataloader):
            _, clip_feat, roi, img_idx = fdata
            roi = roi.to(device, dtype=torch.float32)
            clip_feat = clip_feat.to(device, dtype=torch.float32)
            outs = model(roi[:, 0, :])
            temp = []
            for out in outs:
                temp.append(out.cpu().detach().numpy())
            base_clip.append(temp)

            for i in range(k):
                mean_r[i].append(batch_r2(outs[i], clip_feat[:, i, :]))
                mean_pcc[i].append(
                    batch_pcc(outs[i].cpu().detach().numpy(), clip_feat[:, i, :].cpu().detach().numpy()))
            loss0 = criterion1(clip_feat[:, 0, :], outs[0])
            loss1 = criterion1(clip_feat[:, 1, :], outs[1])
            loss2 = criterion1(clip_feat[:, 2, :], outs[2])
            loss = loss0 + loss1 + loss2
            valid_losses.append(loss.item())
            # print(loss)
            progress.set_postfix({"loss1": loss1.item(), "loss0": loss0.item()})
            progress.update()
        base_clip = np.array(base_clip)
        base_clip = base_clip[:, :, 0, :]
        # print(base_clip.shape)
        if epoch == 0 or (epoch+1) % args.save_every == 0 or epoch == epochs - 1:
            # if base_clip.shape[-1] == 768:
            #     np.save(f'Decoded_clip_{subj}_pami/clip_Biinfo_{weight}_ckpt{epoch}_test{t}_{seed}_tem{tem}.npy',
            #             base_clip)
            # elif base_clip.shape[-1] == 77 * 768:
            #     np.save(f'Decoded_clip_{subj}_pami/c_Biinfo_{weight}_ckpt{epoch}_test{t}_{seed}.npy', base_clip)
            for i in range(k):
                r = sum(mean_r[i]) / len(mean_r[i])
                pcc = sum(mean_pcc[i]) / len(mean_pcc[i])
                print(f'r{i}:', r, f'pcc{i}:', pcc)
        model.train()
        progress.close()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=5)
    # parser.add_argument('--subj', type=str, default='subj01')
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='./All_subj01_early2')
    weight_list = [0.01]  # [0,0.01,0.05,0.1,0.5,1]
    args = parser.parse_args()
    out_dir = args.out_dir
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    k = args.k
    gpu = args.gpu
    # seed = args.seed
    os.makedirs(args.out_dir, exist_ok=True)
    subjs = ['subj01']
    seeds =  [159]
    # subj = args.subj
    test = 1
    # device = torch.device(f'cuda:{gpu}')
    # print(args)
    # kwargs = vars(args)
    # print(type(kwargs))
    # dataset = zero_dataset('train',k = k)
    # dataset.set_dataset('train')
    test_num = 500
    for subj in subjs:
        for seed in seeds:
            dataset = pretrain_dataset('train', subj=subj, test_num=test_num, k=k, seed=seed)
            for t in range(test):
                for weight in weight_list:
                    train_cons(args, dataset, lr, bs, gpu, out_dir, k, weight, t, seed, subj)
    # train_diff(args,dataset,lr,bs,gpu,out_dir,k)
    # dataset = pretrain_dataset('train',k=k)
    # train_all(args,dataset,lr,bs,gpu,out_dir,k)
