import clip
import numpy as np
import torch
from Analysis.nsd_access import NSDAccess
from tqdm import tqdm
import PIL
import os


def seimg_clip(session_cnt=1, subject='subj01'):
    nsd_path = '/data1/rzxia/Project/StableDiffusionReconstruction/nsd'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    clip_model, process = clip.load('ViT-L/14', device=device, jit=False)
    nsda = NSDAccess(nsd_path)
    session_img = []
    beh = nsda.read_behavior(subject=subject,
                             session_index=session_cnt)

    stims_all = beh['73KID'] - 1
    print(stims_all.values)
    os.makedirs(f'alldata_{subject}/img_clip(L)', exist_ok=True)
    for s in tqdm(stims_all.values):
        img = nsda.read_images(s)
        pil_image = PIL.Image.fromarray(img)
        processimg = process(pil_image).unsqueeze(0).to(device)
        img_feat = clip_model.encode_image(processimg).flatten().detach().cpu().numpy()
        # print(img_feat.shape)
        session_img.append(img_feat)
    session_img = np.array(session_img)
    print(session_img.shape) #(750,768)
    np.save(f'alldata_{subject}/img_clip(L)/session_{session_cnt}_clip.npy',session_img)




if __name__ == '__main__':
    session_num = 38
    for i in range(1, session_num):

        seimg_clip(session_cnt=i,subject='subj01')
        # seimg_clip(session_cnt=i, subject='subj07')


