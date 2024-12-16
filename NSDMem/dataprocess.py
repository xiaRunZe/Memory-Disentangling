import scipy
import pandas as pd
import numpy as np
import os

from Analysis.nsd_access import NSDAccess
def processMem_img(subj='subj01',atlasname='streams'):
    nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'
    behavpath = f'{nsd_path}/nsddata/ppdata/{subj}/behav/responses.tsv'
    subject = subj
    # behav_tsv =  pd.read_csv(behavpath,sep='\t')
    nsd_expdesign = scipy.io.loadmat(f'{nsd_path}/nsddata/experiments/nsd/nsd_expdesign.mat')
    nsda = NSDAccess(nsd_path)
    # columns = behav_tsv.columns
    # len_tsv = len(behav_tsv)
    session_cnt=37
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    print(atlas[1])
    print(atlas[0].shape) #(81,104,83)

    behs = pd.DataFrame()
    for i in range(1, session_cnt):
        beh = nsda.read_behavior(subject=subject,session_index=i)
        behs = pd.concat((behs, beh))
    stims_all = behs['73KID'] - 1
    print(behs.columns)
    # print(stims_all)
    # print(behs.columns)
    os.makedirs(f'alldata_{subj}/sedata',exist_ok=True)
    for i in range(1,session_cnt):

        print('session:',i)
        beta_trial = nsda.read_betas(subject=subject,
                                session_index=i,
                                trial_index=[], # empty list as index means get all for this session
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
        print(beta_trial.shape)#(750,83,104,81)  单session的所有全脑数据
        print(beta_trial[0])
        np.save(f'alldata_{subj}/sedata/session_{i}_fmri.npy',beta_trial)

        # if i==1:
        #     betas_all = beta_trial.get_fdata().transpose([3,0,1,2])
        # else:
        #     betas_all = np.concatenate((betas_all,beta_trial.get_fdata().transpose([3,0,1,2])),0)
        # for roi, val in atlas[1].items():
    # print(betas_all.shape)
    # betas_roi = betas_all[:, atlas[0].get_fdata() == 1]
    # print(betas_roi.shape)



if __name__ == '__main__':

    atlasname = 'nsdgeneral'
    # atlasname = 'Kastner2015'
    # atlasname = 'HCP_MMP1'
    # atlasname = 'streams'
    # atlasname = 'corticalsulc'
    processMem_img(f'subj01',atlasname=atlasname)

