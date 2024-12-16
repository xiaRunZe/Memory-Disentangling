import argparse
import numpy as np
from neurora.rdm_cal import fmriRDM_roi
import os
import sys
sys.path.append('../')
from nsd_access import NSDAccess


def seall_RDM(session_idx,useroi=False,subject='subj01'):
    data_path = f'../alldata/sedata/session_{session_idx}_fmri.npy'
    session_data = np.load(data_path)
    # print(session_data)
    trails_num = session_data.shape[0]
    nx, ny, nz = session_data.shape[1:]
    print(nx,ny,nz)
    if useroi:
        nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'
        nsda = NSDAccess(nsd_path)
        atlasname = 'streams'
        atlas,mask_idx= nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm') #template
        print(mask_idx)
        print(atlas.shape)  # (81,104,83)
        print(type(atlas))
        atlas = atlas.get_fdata().transpose(2,1,0)
        print(type(atlas))
    session_data = np.reshape(session_data,[trails_num,1,nx,ny,nz])
    # for i in range(trails_num):
    #     for  j in range(trails_num):
    #         if i==j:
    #           break;
    #           nps[i][j] =  nps_fmri(session_data[[i,j]])
    # print(nps)
    rdm = fmriRDM_roi(session_data,atlas)
    print(rdm.shape)
    np.save(f'fmri_RDM/session{session_idx}_rdm.npy',rdm)

def seroi_RDM(session_idx,subject='subj01'):
    data_path = f'../alldata_{subject}/sedata/session_{session_idx}_fmri.npy'
    session_data = np.load(data_path)
    # print(session_data)
    trails_num = session_data.shape[0]
    nx, ny, nz = session_data.shape[1:]
    print(nx, ny, nz)
    nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'
    nsda = NSDAccess(nsd_path)
    atlasname = 'streams'
    atlasname = 'nsdgeneral'
    roi='nsdgeneral'
    atlas, mask_idx = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    print(atlas.shape)  # (81,104,83)
    roi_idx = mask_idx[roi]
    atlas = atlas.get_fdata().transpose(2, 1, 0)
    atlas = (atlas==roi_idx).astype(int)
    print(atlas.shape)
    session_data = np.reshape(session_data, [trails_num, 1, nx, ny, nz])
    rdm = fmriRDM_roi(session_data, atlas)
    print(rdm.shape)
    np.save(f'{subject}/roi_RDM/session{session_idx}_{roi}_rdm.npy', rdm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_begin',type=int,default=1)
    parser.add_argument('--session_end',type=int,default=38)
    parser.add_argument('--useroi',type=bool,default=True)
    args = parser.parse_args()
    session_begin = args.session_begin
    session_end = args.session_end
    useroi = args.useroi

    for i in range(session_begin, session_end):
        seroi_RDM(i,subject='subj01')