import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rcParams



def trailwise_sim(session_idx,k=0,use_roi=False,subj='subj01'  , roi = 'ventral'):

    # roi = 'nsdgeneral'
    if use_roi:
        rdm_path = f'{subj}/roi_RDM/session{session_idx}_{roi}_rdm.npy' #1,trails,
    else:
        rdm_path = f'{subj}/fmri_RDM/session{session_idx}_rdm.npy'  # 1,trails,
    clip_path = f'{subj}/clip_RDM/session{session_idx}_cliprdm.npy'

    rdm = np.load(rdm_path)
    clip_rdm = np.load(clip_path)
    trails_num = rdm.shape[-1]

    rdm = np.squeeze(rdm)
    clip_rdm = np.squeeze(clip_rdm)
    # clip_path2 = f'roi_RDM/session{session_idx}_rdm.npy'
    # clip_rdm2 = np.load(clip_path2)
    # clip_rdm2 = clip_rdm2.reshape(trails_num, trails_num)

    # print(rdm)
    # print(clip_rdm)
    test = []
    trails_r = []
    p_r = []
    p_test = []
    max_k =10
    trails_num = trails_num - max_k
    for i in range(trails_num):
        r,p =pearsonr(rdm[i+k],clip_rdm[i])
        trails_r.append(r)
        p_r.append(p)
        r2,p2 = np.corrcoef(rdm[i+k],rdm[i])
        test.append(r2)
        p_test.append(p2)
    trails_r = np.array(trails_r)
    test = np.array(test)
    p_r = np.array(p_r)
    p_test = np.array(p_test)
    # print(f'P:session{session_idx}_k{k}',p_r.mean())
    # print(f'P:testclip{session_idx}_k{k}', p_test.mean())

    return trails_r.mean(),test.mean()



if __name__ == '__main__':

    k_max = 10
    x1 = range(k_max)
    x2 = range(1,k_max)
    use_roi = True
    subjs = ['subj01','subj02','subj05' ,'subj07']
    roi = 'nsdgeneral'
    session_num = 37
    allsub_bcr = []
    allsub_bbr = []
    for subj in subjs:
        all_bcr = np.zeros([k_max,session_num])
        all_bbr = np.zeros([k_max-1,session_num])
        all_bcr_abs = np.zeros([k_max, session_num])
        all_bbr_abs = np.zeros([k_max - 1, session_num])
        for i in tqdm(range(1,session_num+1)):
            bcr = []
            bbr = []
            bcr_abs = []
            bbr_abs = []
            for k in range(k_max):
                r1,r2 = trailwise_sim(i,k,use_roi,subj,roi)
                bcr.append(r1)
                bcr_abs.append(abs(r1))
                all_bcr[k][i-1] = r1
                all_bcr_abs[k][i - 1] = abs(r1)
                if k!=0:
                    bbr.append(r2)
                    bbr_abs.append(abs(r2))
                    all_bbr[k-1][i-1] = r2
                    all_bbr_abs[k-1][i-1] = abs(r2)

        ave_bcr = []
        ave_bbr = []
        ave_bcr_abs = []
        ave_bbr_abs = []
        for i in range(k_max):
            ave_bcr.append(all_bcr[i].mean())
            ave_bcr_abs.append(all_bcr_abs[i].mean())
            if i!=(k_max-1):
                ave_bbr.append(all_bbr[i].mean())
                ave_bbr_abs.append(all_bbr_abs[i].mean())
        allsub_bbr.append(ave_bbr)
        allsub_bcr.append(ave_bcr)
    # plt.title('average_roi' if use_roi else 'average_all')
    # plt.plot(x1, ave_bcr, x2, ave_bbr)
    # plt.legend(['Brain-Image Simi', 'Brain-Time Simi'])
    # plt.show()
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
    plt.figure()
    # plt.title(f'Trail-wise RSA Results Between CLIP Embedding and fMRI Voxels')
    for i, subj in enumerate(subjs):
        plt.plot(x1, allsub_bcr[i], marker='o', label=f'{subj}')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("Trail-wise RSA Score")
    plt.savefig(f'all_subjects_RSA_bcr{roi}.pdf',format='pdf')
    plt.clf()
    plt.figure()
    # plt.title(f'Trail-wise RSA Results Between Current and Past fMRI Voxels')
    for i, subj in enumerate(subjs):
        plt.plot(x2, allsub_bbr[i], marker='o', label=f'{subj}')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("Trail-wise RSA Score")
    plt.savefig(f'all_subjects_RSA_bbr{roi}.pdf',format='pdf')
    plt.clf()

    # plt.plot(x1[3:], ave_bcr[3:], marker='o')
    # plt.legend(['Brain-Image Simi'])
    # plt.savefig(f'{subj}/Brain-{roi} Simi(RSA)(3-n).png')
    # plt.clf()
    # plt.title(f'average_bbs_{roi}' if use_roi else 'average_bbs_all')
    # plt.plot(x2, ave_bbr,marker='o')
    # plt.legend(['Brain-Time Simi'])
    # plt.savefig(f'{subj}/Brain-{roi} time Simi(RSA).png')
    # plt.show()
    # plt.clf()
    #
    #
    # for i, y in enumerate(all_bcr):
    #     if i <= 1 :
    #         continue
    #     plt.plot( range(len(y)),y, label=f'L{i}')
    # plt.legend()
    # plt.show()
    #
    # for i, y in enumerate(all_bbr):
    #     if i<=1:
    #         continue
    #     plt.plot( range(len(y)),y, label=f'L{i+1}')
    # plt.legend()
    # plt.show()