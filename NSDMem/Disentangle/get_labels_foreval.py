import os
import sys
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from fmridataset import fmri_dataset
import torch.nn.functional as F
from pretrain_zerodata import pretrain_dataset
import argparse
from basemodel import base_disentangle
from MLP_train import  MLP
from transformers import AdamW, get_linear_schedule_with_warmup
from All_model import all_model,all_model_diff,all_model_diff2
from scipy.stats import pearsonr
import sys
sys.path.append('../')
from Analysis.nsd_access import NSDAccess

def batch_pcc(predictions, targets):
    batch_pcc = []
    batch_p = []
    for i in range(predictions.shape[0]):
        corr_matrix,p = pearsonr(predictions[i], targets[i])
        batch_pcc.append(corr_matrix)
        batch_p.append(p)
    return sum(batch_pcc)/len(batch_pcc)
subjs = ['subj01','subj02','subj05','subj07']
seeds = [789,159,233,265,357,]
k = 3
gpu = 0
test_num = 500


for subj in subjs:
    for seed in seeds:
        target_dir = f'decoded/{subj}'
        os.makedirs(f'{target_dir}/captions_{seed}',exist_ok=True)
        os.makedirs(f'{target_dir}/images_{seed}',exist_ok=True)

        device = torch.device(f'cuda:{gpu}')
        dataset = pretrain_dataset('train',subj=subj,k=k,test_num=test_num,seed=seed)
        train_dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        print(len(train_dataloader))
        print('dataloader',len(train_dataloader))
        _, clip_feat, roi, img_idx = dataset[0]
        roi_dim = roi.shape[-1]


        temp = []
        train_clip = []
        test_clip = []
        train_roi = []
        test_roi = []

        for idx, fdata in enumerate(train_dataloader):
            _, clip_feat, roi, img_idx=fdata
            train_clip.append(clip_feat[0].detach().numpy())
            train_roi.append(roi[0].detach().numpy())
        train_roi = np.array(train_roi)
        train_clip = np.array(train_clip)
        print(train_roi.shape)
        np.save(f'../Analysis/{subj}/train_roi_{seed}_{k}.npy',train_roi)
        np.save(f'../Analysis/{subj}/train_clip_{seed}_{k}.npy',train_clip)

        for idx, fdata in enumerate(dataloader):
            _, clip_feat, roi, img_idx = fdata
            test_clip.append(clip_feat[0].detach().numpy())
            test_roi.append(roi[0].detach().numpy())
        test_roi = np.array(test_roi)
        test_clip = np.array(test_clip)
        print(test_roi.shape)
        np.save(f'../Analysis/{subj}/test_roi_{seed}_{k}.npy',test_roi)
        np.save(f'../Analysis/{subj}/test_clip_{seed}_{k}.npy',test_clip)
        # #base
        mlp_clip = []
        gt_clip= []
        for idx, fdata in enumerate(dataloader):
            _, clip_feat, roi, img_idx = fdata
            roi = roi.to(device, dtype=torch.float32)
            temp = []
            # for i in range(k):
                # out = temp_model[i](roi)
                # temp.append(out.cpu().detach().numpy())
            # mlp_clip.append(temp)
            gt_clip.append(clip_feat[0].detach().numpy())
        gt_clip = np.array(gt_clip)
        # print(gt_clip.shape)
        # print(batch_pcc(base_clip[:,1,:],gt_clip[:,1,:]))
        # print(batch_pcc(mlp_clip[:,1,:],gt_clip[:,1,:]))
        # np.save(f'{target_dir}/mlp_clip{ckpt_epoch}.npy',mlp_clip) #n,k,768
        np.save(f'{target_dir}/gt_clip_{seed}.npy',gt_clip)


        #get captions
        nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'

        nsda = NSDAccess(nsd_path)
        captions = [[] for i in range(k)]
        idx_k  = [[] for i in range(k)]
        for idx,fdata in tqdm(enumerate(train_dataloader)):
            _, clip_feat, roi, img_idx = fdata
            roi = roi.to(device,dtype=torch.float32)
            # print(img_idx)
            for i in range(k):
                prompt = []
                # print(img_idx[0][i])
                idx_k[i].append(img_idx[0][i].item())
                # prompts = nsda.read_image_coco_info([img_idx[0][i].item()], info_type='captions')
                # for p in prompts:
                #     prompt.append(p['caption'])
                #
                #     break
                # captions[i].append(prompt[0])
        idx_k = np.array(idx_k,dtype=int)
        np.save(f'{target_dir}/captions_{seed}/3stage_imgidxs_tr.npy',idx_k)
        print(idx_k.shape)
        for i in range(k):
            # print(captions[i])
            # break
            # print(idx_k[i])
            prompts = nsda.read_image_coco_info(idx_k[i], info_type='captions')
            for num,idx in enumerate(idx_k[i]):
                img  = nsda.read_images(idx)
                image = Image.fromarray(img)
                # 保存图像
                image.save(f'{target_dir}/images_{seed}/{num}_{i}.png')
                # print(img.shape)
            # print(len(prompts[0]))
            prompt = []
            for p in prompts:
                # print(p)
                for item in p:
                    prompt.append(item['caption'])
                    break
            print(len(prompt))
            # captions[i].append(prompt)
            # df = pd.DataFrame(prompt)
            # df.to_csv(f'{target_dir}/coco_captions_brain_{i}.csv', sep='\t', header=False, index=False)
            output_file = f'{target_dir}/captions_{seed}/coco_captions_brain_{i}_tr.txt'
            output_csv = f'{target_dir}/captions_{seed}/coco_captions_brain_{i}_tr.csv'
            df = pd.DataFrame(prompt)
            df.to_csv(output_csv, index=False, header=False, sep='\t',encoding='utf-8')

            with open(output_file, 'w', encoding='utf-8') as f:
                for sentence in prompt:
                    clean_sentence = sentence.rstrip('\n')
                    f.write(clean_sentence + '\n')

        dataset.set_dataset('test')
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        captions = [[] for i in range(k)]
        idx_k = [[] for i in range(k)]
        for idx, fdata in tqdm(enumerate(dataloader)):
            _, clip_feat, roi, img_idx = fdata
            roi = roi.to(device, dtype=torch.float32)
            # print(img_idx)
            for i in range(k):
                prompt = []
                idx_k[i].append(img_idx[0][i].item())
        idx_k = np.array(idx_k,dtype=int)
        np.save(f'{target_dir}/captions_{seed}/3stage_imgidxs_te.npy',idx_k)
        print(idx_k.shape)

        for i in range(k):
            print(idx_k[i])
            prompts = nsda.read_image_coco_info(idx_k[i], info_type='captions')
            prompt = []
            for p in prompts:
                # print(p)
                for item in p:
                    prompt.append(item['caption'])
                    break
            print(len(prompt))
            output_file = f'{target_dir}/captions_{seed}/coco_captions_brain_{i}_te.txt'
            output_csv = f'{target_dir}/captions_{seed}/coco_captions_brain_{i}_te.csv'
            with open(output_file, 'w', encoding='utf-8') as f:
                for sentence in prompt:
                    clean_sentence = sentence.replace('\n', '').strip()
                    if clean_sentence:
                        f.write(clean_sentence + '\n')
            df = pd.DataFrame(prompt)
            df.to_csv(output_csv, index=False, header=False, encoding='utf-8')
