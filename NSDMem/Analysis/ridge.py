import sys
import numpy as np
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
sys.path.append('../')
from Analysis.nsd_access import NSDAccess
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from matplotlib import rcParams


def idx2sessidx(index,data_num):
    session_idx = int(index / data_num)
    inner_idx = index % data_num
    return session_idx,inner_idx

def batch_r2(predictions, targets):
    batch_r2 = []
    for i in range(predictions.shape[0]):
        ss_tot = np.sum((targets[i] - targets[i].mean()) ** 2)
        ss_res = np.sum((predictions[i] - targets[i]) ** 2)
        if ss_tot == 0:
            r2 = 0
        else:
            r2 = 1 - ss_res / ss_tot
        batch_r2.append(r2)
    return sum(batch_r2) / len(batch_r2)

def batch_pcc(predictions, targets):
    batch_pcc = []
    batch_p = []
    for i in range(predictions.shape[0]):
        corr_matrix,p = pearsonr(predictions[i], targets[i])
        batch_pcc.append(corr_matrix)
        batch_p.append(p)
    return sum(batch_pcc)/len(batch_pcc),sum(batch_p)/len(batch_p)

def ridge_ana(subj= 'subj07',  alph = 1e6):
    roi_session = []
    clip_session = []
    idx_seeion = []
    data_path = f'/data1/rzxia/Project/NsdMem/alldata_{subj}'
    session_cnt = 37

    nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'
    nsda = NSDAccess(nsd_path)
    behs = pd.DataFrame()
    stims_sess = []
    for i in range(1, session_cnt + 1):
        beh = nsda.read_behavior(subject=subj, session_index=i)
        behs = pd.concat((behs, beh))
        stims_sess.append(beh['73KID'].to_numpy() - 1)  #
    # print(self.stims_sess)
    stims_sess = np.array(stims_sess)
    print(stims_sess.shape)
    # atlasname = 'streams'
    # roi_list = ['ventral']
    # atlasname = 'HCP_MMP1'  # 106619
    # roi_list = ['V1', 'MST', 'V2', 'V3', 'V4', 'V3A', 'V3B', 'MT', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'LIPv', 'LIPd']  # 14019
    atlasname = 'nsdgeneral'
    roi_list = ['nsdgeneral']
    atlas, mask_idx = nsda.read_atlas_results(subject=subj, atlas=atlasname, data_format='func1pt8mm')


    # roi_idx = mask_idx[roi]
    for sessidx in tqdm(range(session_cnt)):
        betas_voxel = []
        session_path = f'{data_path}/sedata/session_{sessidx+1}_fmri.npy'
        clip_path = f'{data_path}/img_clip(L)/session_{sessidx+1}_clip.npy'
        # clip_path = f'{data_path}/session_c/session_{sessidx+1}_c.npy'
        now_sessfmri = np.load(session_path).astype("float32")
        now_sessclip = np.load(clip_path).astype("float32")
        for roi in roi_list:
            roi_idx = mask_idx[roi]
            betas_voxel.append(now_sessfmri[:, atlas.get_fdata().transpose([2, 1, 0]) == roi_idx])
        betas_roi = np.hstack(betas_voxel)
        # betas_roi = now_sessfmri[:, atlas.get_fdata().transpose([2, 1, 0]) == roi_idx]
        roi_session.append(betas_roi)
        clip_session.append(now_sessclip)
    roi_session = np.array(roi_session)
    clip_session = np.array(clip_session)
    # roi_session = np.vstack(roi_session)
    # clip_session = np.vstack(clip_session)
    # roi_session = roi_session.reshape(roi_session.shape[0]*roi_session.shape[1],roi_session.shape[2])
    # clip_session = clip_session.reshape(clip_session.shape[0]*clip_session.shape[1],clip_session.shape[2])
    # print(roi_session.shape) #36,750,7604
    # print(clip_session.shape) #36,750,768
    max_k = 10

    data_num = 750-max_k+1
    seed = 42
    np.random.seed(seed)
    test_num = 4000
    test_indices = np.random.choice(data_num*session_cnt, size=test_num, replace=False)
    test_indices= np.sort(test_indices)

    pcc_list = []
    r2_list = []
    begin_k = 1
    k_list = list(range(begin_k-1,max_k))
    k_list = [0] + [i for i in k_list[1:]] + ['rand']
    test_idxset = []
    for idx in test_indices:  # test img idx
        seidx, inidx = idx2sessidx(idx, data_num)
        # roi = roi_session[seidx][inidx+k-1]
        # clip = clip_session[seidx][inidx]
        clip_idx = stims_sess[seidx][inidx]
        if clip_idx not in test_idxset:
            test_idxset.append(clip_idx)
    for k in range(begin_k,max_k+1): #split train and test
        test_roi = []
        test_clip = []
        test_idx = []
        train_roi = []
        train_clip= []
        train_idx = []
        for seidx in range(session_cnt):
            for inidx in range(data_num):
                this_roi = roi_session[seidx][inidx+k-1]
                this_clip = clip_session[seidx][inidx]
                this_clip_idx = stims_sess[seidx][inidx]
                if this_clip_idx not in test_idxset: #trainset
                    train_roi.append(this_roi)
                    train_clip.append(this_clip)
                    train_idx.append(this_clip_idx)
                else:
                    test_roi.append(this_roi)
                    test_clip.append(this_clip)
                    test_idx.append(this_clip_idx)
        train_roi = np.array(train_roi)
        train_clip = np.array(train_clip)
        train_idx = np.array(train_idx)
        test_roi = np.array(test_roi)
        test_clip = np.array(test_clip)
        test_idx = np.array(test_idx)
        norm_mean_train = np.mean(train_roi, axis=0)
        norm_scale_train = np.std(train_roi, axis=0, ddof=1)
        train_roi = (train_roi - norm_mean_train) / norm_scale_train
        test_roi = (test_roi - norm_mean_train) / norm_scale_train

        # print(np.mean(train_roi), np.std(train_roi))
        # print(np.mean(test_roi), np.std(test_roi))
        #
        # print(np.max(train_roi), np.min(train_roi))
        # print(np.max(test_roi), np.min(test_roi))
        reg = skl.Ridge(alpha=alph)
        reg.fit(train_roi, train_clip)
        pred_clip = reg.predict(test_roi)
        # pred_test_latent = reg.predict(test_roi)
        # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / np.std(pred_test_latent, axis=0)
        # pred_clip = std_norm_test_latent * np.std(train_clip, axis=0) + np.mean(train_clip, axis=0)
        # pcc = np.corrcoef(test_clip.flatten(), pred_clip.flatten())[0, 1:]
        pcc,p_value = batch_pcc(pred_clip,test_clip)
        r2 = reg.score(test_roi, test_clip)
        r2 = batch_r2(pred_clip, test_clip)
        r2_list.append(r2)
        pcc_list.append(pcc)
        print(f'k:{k-1},pcc:{pcc},p_value:{p_value},r2:{r2}')
        # print(train_roi.shape,train_clip.shape,train_idx.shape)
        # print(test_roi.shape,test_clip.shape,test_idx.shape)



    def get2idx(se_cnt,data_n):
        idx1 = np.random.choice(se_cnt, 1)
        idx2 = np.random.choice(data_n, 1)
        idx1 = idx1[0] if idx1.ndim else idx1
        idx2 = idx2[0] if idx2.ndim else idx2
        return idx1,idx2
    all_datanum = data_num * session_cnt
    all_data = []
    now_DT = []  #存2元下标组
    print(all_datanum)
    for idx in tqdm(range(all_datanum)):
        seidx, inidx = idx2sessidx(idx, data_num)
        roi = roi_session[seidx][inidx]
        while(True):
            rand_seidx,rand_inidx = get2idx(session_cnt,data_num)
            if (rand_seidx,rand_inidx) not in now_DT:
                now_DT.append((rand_seidx,rand_inidx))
                # print((rand_seidx,rand_inidx))
                break
            # else:
            #     print("Repeat!!!",(rand_seidx,rand_inidx))

        clip = clip_session[rand_seidx][rand_inidx]
        clip_idx = stims_sess[rand_seidx][rand_inidx]
        all_data.append((roi,clip,clip_idx))
    print(len(all_data))
    test_roi = []
    test_clip = []
    test_idx = []
    train_roi = []
    train_clip = []
    train_idx = []
    # test_idxset = []
    # for idx in test_indices:
    #     roi,clip,clip_idx = all_data[idx]
    #     if clip_idx not in test_idxset:
    #         test_idxset.append(clip_idx)
    for idx in range(all_datanum):
        roi, clip, clip_idx = all_data[idx]
        if clip_idx in test_idxset:
            test_roi.append(roi)
            test_clip.append(clip)
            test_idx.append(clip_idx)
        else:
            train_roi.append(roi)
            train_clip.append(clip)
            train_idx.append(clip_idx)
    train_roi = np.array(train_roi)
    train_clip = np.array(train_clip)
    train_idx = np.array(train_idx)
    test_roi = np.array(test_roi)
    test_clip = np.array(test_clip)
    test_idx = np.array(test_idx)
    norm_mean_train = np.mean(train_roi, axis=0)
    norm_scale_train = np.std(train_roi, axis=0, ddof=1)
    train_roi = (train_roi - norm_mean_train) / norm_scale_train
    test_roi = (test_roi - norm_mean_train) / norm_scale_train
    #
    # print(np.mean(train_roi), np.std(train_roi))
    # print(np.mean(test_roi), np.std(test_roi))
    #
    # print(np.max(train_roi), np.min(train_roi))
    # print(np.max(test_roi), np.min(test_roi))

    print(train_roi.shape,train_clip.shape,train_idx.shape)
    print(test_roi.shape,test_clip.shape,test_idx.shape)
    reg = skl.Ridge(alpha=alph)
    reg.fit(train_roi, train_clip)
    pred_clip = reg.predict(test_roi)
    # r2 = reg.score(test_roi, test_clip)
    r2 = batch_r2(pred_clip, test_clip)
    r2_list.append(r2)
    # pred_test_latent = reg.predict(test_roi)
    # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / np.std(pred_test_latent, axis=0)
    # pred_clip = std_norm_test_latent * np.std(train_clip, axis=0) + np.mean(train_clip, axis=0)
    # pcc = np.corrcoef(test_clip.flatten(), pred_clip.flatten())[0, 1:]
    pcc,p_value = batch_pcc(pred_clip,test_clip)
    print(pcc,p_value,r2)
    pcc_list.append(pcc)
    return k_list, pcc_list, r2_list
    # print(f'k:{k},pcc:{pcc}')
    #
def plot_results(subjects):
    all_pcc_lists = []
    all_r2_lists = []
    k_list = None
    rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': [8, 6],
        'savefig.dpi': 300,
        'savefig.transparent': True,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.5,
    })
    alph = 1e4
    roi = 'nsdgeneral'
    for subj in subjects:
        k_list, pcc_list, r2_list = ridge_ana(subj,alph)
        all_pcc_lists.append(pcc_list)
        all_r2_lists.append(r2_list)

    plt.figure()
    for i, subj in enumerate(subjects):
        plt.plot(k_list, all_pcc_lists[i], marker='o', label=f'{subj}')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("Pearson correlation coefficient")
    plt.savefig(f'all_subjects_pcc{alph}_{roi}.pdf',format='pdf')
    plt.clf()

    plt.figure()
    for i, subj in enumerate(subjects):
        plt.plot(k_list, all_r2_lists[i], marker='o', label=f'{subj}')
    plt.legend()
    plt.savefig(f'all_subjects_r2{alph}_{roi}.pdf',format='pdf')
    plt.clf()
    '''
    k_list1 = k_list[:3]
    pcc_list1 = pcc_list[:3]
    k_list2 = k_list[3:]
    pcc_list2 = pcc_list[3:]
    plt.plot(k_list, pcc_list,marker='o')
    plt.legend([f'Brain-C Score[{seed}]'])
    plt.savefig(f'{subj}/Brain-C Score[{seed}].png')
    plt.clf()
    plt.plot(k_list, r2_list,marker='o')
    plt.legend([f'Brain-C R2[{seed}]'])
    plt.savefig(f'{subj}/Brain-C R2[{seed}].png')
    plt.clf()
    plt.plot(k_list1, pcc_list1,marker='o')
    plt.legend([f'Brain-C Score[{seed}]'])
    plt.savefig(f'{subj}/Brain-C Score[{seed}](0-2).png')
    plt.clf()
    plt.plot(k_list2, pcc_list2,marker='o')
    plt.legend([f'Brain-C Score[{seed}]'])
    plt.savefig(f'{subj}/Brain-C Score[{seed}](3-n).png')
    plt.clf()
    
    '''


subjs = ['subj01','subj02','subj05','subj07']
plot_results(subjs)
