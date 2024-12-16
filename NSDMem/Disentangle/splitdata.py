import numpy as np
from fmridataset import fmri_dataset
import pickle
from tqdm import tqdm


def split_train(dataset,test_num=300,seed = 42):
    test_set = []
    test_idx = []
    train_set = []
    train_idx = []
    session_cnt = 37
    k = dataset.k
    np.random.seed(seed)
    print(len(dataset))
    test_indices = np.random.choice(len(dataset), size=test_num, replace=False)
    test_indices= np.sort(test_indices)
    # print(test_indices)
    for i in test_indices:
        # print(i)
        fmri, clip_feat, roi,img_idx = dataset[i]
        # print(img_idx)
        test_set.append(dataset[i])
        for idx in img_idx:
            if idx not in test_idx:
                test_idx.append(idx)
    for i in tqdm(range(len(dataset))):
        fmri, clip_feat, roi, img_idx = dataset[i]
        if any(idx in test_idx for idx in img_idx):
            continue
        train_set.append(dataset[i])
        for idx in img_idx:
            if idx not in train_idx:
                train_idx.append(idx)
    print(len(test_set)) #300 400 500 600 700 1000
    print(len(test_idx)) #853 1125 1385 1642 1902 2618
    print(len(train_set)) #21213 19542 18059 16682 15309 11904
    print(len(train_idx)) #8912 8634 8368 8108 7841 7093
    return train_set,test_set  #two list for dataset



#
if __name__ == "__main__":
    dataset = fmri_dataset('all',)
    k = 3
    dataset.set_k(k)
    test_num = 500
    train,test = split_train(dataset=dataset,test_num=test_num)
