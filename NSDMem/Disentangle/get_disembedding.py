import torch
from torch.utils.data import DataLoader
from pretrain_zerodata import pretrain_dataset
import numpy as np
from All_model import all_model
torch.multiprocessing.set_sharing_strategy('file_system')

def get_encodeoutput(dataset,subj='subj01',w=0.01,seed=42,epoch=20):
    test_num = 500
    device = torch.device('cuda:0')
    k = 3
    model_path = f'All_{subj}_early2/{subj}_{w}_{seed}_cons_all-{epoch:03d}.pt'
    print(model_path)
    _, clip_feat, roi, img_idx = dataset[0]
    roi_dim = roi.shape[-1]
    model = all_model(model_type='mlp', in_dim=roi_dim, h=roi_dim // 4, out_dime=roi_dim // 4,
                      out_dimd=clip_feat.shape[-1], n=2
                      , window_k=k)
    model.load_state_dict(torch.load(model_path),strict=True)
    model.to(device)
    dataset.set_dataset('train')
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    now_embeddings_tr = []
    before_embeddings_tr = []
    clip_target_tr = []
    for idx, fdata in enumerate(train_dataloader):
        model.zero_grad()
        _, clip_feat, roi, img_idx = fdata
        roi = roi.to(device, dtype=torch.float32)
        # clip_feat = clip_feat.to(device, dtype=torch.float32)
        # print(clip_feat.shape)
        now_out1, before_out1 = model.encoder(roi[:, 0, :])   #now->clip0  before->clip12
        now_embeddings_tr.append(now_out1[0].cpu().detach().numpy())
        before_embeddings_tr.append(before_out1[0].cpu().detach().numpy())
        clip_target_tr.append(clip_feat[0].cpu().detach().numpy())
    now_embeddings_tr = np.array(now_embeddings_tr)
    before_embeddings_tr = np.array(before_embeddings_tr)
    clip_target_tr = np.array(clip_target_tr)

    print(now_embeddings_tr.shape,before_embeddings_tr.shape,clip_target_tr.shape)
    np.save(f'Encoding_eval2/now_embedding_tr_{subj}_{w}_{seed}_cons_all-{epoch}', now_embeddings_tr)  # (18098,3931)
    np.save(f'Encoding_eval2/before_embedding_tr_{subj}_{w}_{seed}_cons_all-{epoch}', before_embeddings_tr)  # (500,3931)
    np.save(f'Encoding_eval2/cliptarget_tr_{subj}_{w}_{seed}', clip_target_tr)  # 500,3,768
    dataset.set_dataset('test')
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    now_embeddings_te = []
    before_embeddings_te = []
    clip_target_te = []
    for idx, fdata in enumerate(test_dataloader):
        model.zero_grad()
        _, clip_feat, roi, img_idx = fdata
        roi = roi.to(device, dtype=torch.float32)
        now_out1, before_out1 = model.encoder(roi[:, 0, :])
        now_embeddings_te.append(now_out1[0].cpu().detach().numpy())
        before_embeddings_te.append(before_out1[0].cpu().detach().numpy())
        clip_target_te.append(clip_feat[0].cpu().detach().numpy())
    now_embeddings_te = np.array(now_embeddings_te)
    before_embeddings_te = np.array(before_embeddings_te)
    clip_target_te = np.array(clip_target_te)

    print(now_embeddings_te.shape,before_embeddings_te.shape,clip_target_te.shape)
    np.save(f'Encoding_eval2/now_embedding_te_{subj}_{w}_{seed}_cons_all-{epoch}',now_embeddings_te)
    np.save(f'Encoding_eval2/before_embedding_te_{subj}_{w}_{seed}_cons_all-{epoch}',before_embeddings_te)
    np.save(f'Encoding_eval2/cliptarget_te_{subj}_{w}_{seed}',clip_target_te)

def get_sfembedding(dataset,subj='subj01',seed=42,):
    device = torch.device('cuda:0')
    dataset.set_dataset('train')
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    base_roi = []
    for idx, fdata in enumerate(train_dataloader):
        _, clip_feat, roi, img_idx = fdata
        roi = roi.to(device, dtype=torch.float32)
        base_roi.append(roi[0].cpu().detach().numpy())
    base_roi = np.array(base_roi)
    print(base_roi.shape)
    np.save(f'Encoding_eval/roi_tr_{subj}_{seed}',base_roi)
    dataset.set_dataset('test')
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    base_roi_te = []
    for idx, fdata in enumerate(test_dataloader):
        _, clip_feat, roi, img_idx = fdata
        roi = roi.to(device, dtype=torch.float32)
        base_roi_te.append(roi[0].cpu().detach().numpy())
    base_roi_te = np.array(base_roi_te)
    print(base_roi_te.shape)
    np.save(f'Encoding_eval/roi_te_{subj}_{seed}', base_roi_te) #500,3,15724





if __name__=='__main__':
    subj = 'subj01'
    test_num = 500
    seeds = [159]
    # # seeds = [42]
    k=3
    ws = [0.01]
    epochs = []
    for i in range(100):
        if i==0 or (i+1)%5==0:
            epochs.append(i)
    for seed in seeds:
        dataset = pretrain_dataset('train', subj=subj, test_num=test_num, k=k, seed=seed)
        # get_sfembedding(dataset,subj,seed)
        for w in ws:
            for epoch in epochs:
                get_encodeoutput(dataset,subj=subj,w=w,seed=seed,epoch=epoch)
    # for seed in seeds:
    #     dataset = pretrain_dataset('train', subj=subj, test_num=test_num, k=k, seed=seed)
    #     get_sfembedding(dataset,subj,seed)




