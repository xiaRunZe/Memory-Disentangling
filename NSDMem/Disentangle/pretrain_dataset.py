import os
import pandas as pd
import numpy as np
import torch
import numpy
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from Analysis.nsd_access import NSDAccess


class prefmri_dataset(Dataset):
    def __init__(self,split = 'train',subj = 'subj01'):
        super().__init__()
        self.data_path = f'/data1/rzxia/Project/NsdMem/alldata_{subj}'
        self.now_stage = 1
        self.k = 1   #windows_size

        self.split = split
        if split == 'train':
            self.all_session_num = 33
            self.now_sessidx = 0
        elif split == 'test':
            self.all_session_num = 3
            self.now_sessidx = 33
        elif split == 'all':
            self.all_session_num = 37
            self.now_sessidx = 0
        self.session_path = f'{self.data_path}/sedata/session_{self.now_sessidx+1}_fmri.npy'
        self.clip_path = f'{self.data_path}/img_clip(L)/session_{self.now_sessidx+1}_clip.npy'
        self.c = f'{self.data_path}/session_c/session_{self.now_sessidx + 1}_c.npy'
        self.now_sessfmri = np.load(self.session_path)  # (750,83,104,81)
        self.now_sessclip = np.load(self.clip_path)
        self.now_c = np.load(self.c)
        self.trails_num1 = 63
        self.trails_num2 = 62
        self.all_trails_num = self.now_sessfmri.shape[0]
        self.data_num = self.all_trails_num
        self.data_trails1 = self.trails_num1
        self.data_trails2 = self.trails_num2
        subject = subj
        session_cnt = 37
        nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'
        nsda = NSDAccess(nsd_path)
        behs = pd.DataFrame()
        self.stims_sess = []
        for i in range(1, session_cnt+1):
            beh = nsda.read_behavior(subject=subject, session_index=i)  # 读一个session的数据
            behs = pd.concat((behs, beh))  # 行为的tsv
            self.stims_sess.append(beh['73KID'].to_numpy() - 1)     # 按照trails顺序的图像id，
        # print(self.stims_sess)
        self.stims_sess = np.array(self.stims_sess)
        print(self.stims_sess.shape)
        nsda = NSDAccess(nsd_path)
        # atlasname = 'nsdgeneral'
        # atlasname = 'Kastner2015'
        # atlasname = 'HCP_MMP1'
        # # atlasname = 'streams'
        # # atlasname = 'corticalsulc'
        # roi = 'ventral'
        # self.roi_list = ['V1', 'MST', 'V2', 'V3', 'V4', 'V3A', 'V3B', 'MT', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'LIPv',
        #                  'LIPd']  # 14019
        atlasname = 'nsdgeneral'
        self.roi_list = ['nsdgeneral']
        self.atlas, self.mask_idx = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')

        self.betas_voxel = []
        for roi in self.roi_list:
            roi_idx = self.mask_idx[roi]
            self.betas_voxel.append(self.now_sessfmri[:, self.atlas.get_fdata().transpose([2, 1, 0]) == roi_idx])
        self.betas_voxel = np.hstack(self.betas_voxel)
        print(self.betas_voxel.shape)
        self.roi_idx = self.mask_idx[roi]
        self.betas_roi = self.now_sessfmri[:, self.atlas.get_fdata().transpose([2, 1, 0]) == self.roi_idx]
        # print(betas_roi.shape)
    # def set_stage(self,s):
    #     self.now_stage = s
    #     if s == 1 :
    #         self.k = 1
    #     elif s == 2 :
    #         self.k = 3
    #     elif s ==3 :
    #         self.k = 3
    #     else:
    #         print('Invalid S!!!')
    #     self.data_num = self.trails_num - self.k + 1
    def set_k(self,k):
        self.k = k
        self.data_trails1 = self.trails_num1-self.k+1
        self.data_trails2 = self.trails_num2-self.k+1
        self.data_num = (6 * (self.data_trails1+self.data_trails2))


    def __len__(self):
        return self.all_session_num * self.data_num

    def __getitem__(self, index):
        session_idx = int(index/self.data_num)
        inner_idx = index % self.data_num
        if session_idx != self.now_sessidx:
            self.now_sessidx = session_idx
            # print('Next session:',self.now_sessidx)
            self.session_path = f'{self.data_path}/sedata/session_{self.now_sessidx+1}_fmri.npy'
            self.clip_path = f'{self.data_path}/img_clip(L)/session_{self.now_sessidx+1}_clip.npy'
            self.c = f'{self.data_path}/session_c/session_{self.now_sessidx + 1}_c.npy'
            self.now_sessfmri = np.load(self.session_path)
            self.now_sessclip = np.load(self.clip_path)
            self.now_c = np.load(self.c)
            self.betas_voxel = []
            for mask in self.roi_list:
                roi_idx = self.mask_idx[mask]
                # print(roi_idx)
                self.betas_voxel.append(self.now_sessfmri[:, self.atlas.get_fdata().transpose([2, 1, 0]) == roi_idx])
            self.betas_voxel = np.hstack(self.betas_voxel)
            self.betas_roi = self.now_sessfmri[:, self.atlas.get_fdata().transpose([2, 1, 0]) == self.roi_idx]
        trails_idx = int(inner_idx/(self.data_trails1+self.data_trails2))  #第几个大trails（125一组的）
        trails_inneridx = inner_idx%(self.data_trails1+self.data_trails2)
        if trails_idx==6:
            print(inner_idx)
        if trails_inneridx <=self.data_trails1:  #63那一组
            inner_idx = trails_idx * (self.trails_num1+self.trails_num2) + trails_inneridx
        elif trails_inneridx>self.data_trails1 and trails_inneridx<=(self.data_trails1+self.data_trails2):
            inner_idx = trails_idx * (self.trails_num1+self.trails_num2) + trails_inneridx + self.k-1
        else:
            print('ERROR:INDEX OVER FLOWER')
        this_idx = np.flip(self.stims_sess[self.now_sessidx][inner_idx:inner_idx+self.k],axis=0).copy()  #dataloader不允许负步长
        this_fmri = np.flip(self.now_sessfmri[inner_idx:inner_idx+self.k],axis=0).copy()
        this_clip = np.flip(self.now_sessclip[inner_idx:inner_idx+self.k],axis=0).copy()
        this_c = np.flip(self.now_c[inner_idx:inner_idx + self.k], axis=0).copy()
        # this_roi = np.flip(self.betas_roi[inner_idx:inner_idx+self.k],axis=0).copy()
        this_roi = np.flip(self.betas_voxel[inner_idx:inner_idx+self.k],axis=0).copy()
        # this_fmri = this_fmri.transpose(1,2,3,0)   #这是因为要用AE  最后一纬度是T
        # this_clip = this_clip.transpose(1,2,3,0)
        # this_fmri = np.expand_dims(this_fmri, axis=0)
        # this_clip = np.expand_dims(this_clip, axis=0)
        # print(this_clip.shape,this_roi.shape)
        # if trails_idx==0:
        #     print('aaaa')
        if this_clip.shape[0]==0:
            print(index,trails_idx,inner_idx,self.k)
        return this_fmri,this_clip,this_roi,this_idx    #返回总是切片，k=1时也是 (k,xxx)
    def get_input_shape(self):
        shape = self.now_sessfmri[0].shape
        return shape



