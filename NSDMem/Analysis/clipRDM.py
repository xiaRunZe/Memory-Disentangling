import numpy as np
from scipy.stats import pearsonr
import os



def se_img_RDM(session_idx,subj='subj01'):
    data_path = f'../alldata_{subj}/img_clip(L)/session_{session_idx}_clip.npy'
    session_clip= np.load(data_path)
    trailnum = session_clip.shape[0]
    rdm = np.zeros([trailnum,trailnum])
    os.makedirs(f'{subj}/clip_RDM',exist_ok=True)
    for i in range(trailnum):
        for j in range(trailnum):
            rdm[i][j] = pearsonr(session_clip[i],session_clip[j])[0]
    rdm = 1 - rdm
    print(rdm.shape)
    print(rdm)
    np.save(f'{subj}/clip_RDM/session{session_idx}_cliprdm.npy',rdm)




if __name__ == '__main__':
    session_begin = 1
    session_end = 38
    # subj = 'subj02'
    for i in range(session_begin,session_end):

        se_img_RDM(i,'subj01')
