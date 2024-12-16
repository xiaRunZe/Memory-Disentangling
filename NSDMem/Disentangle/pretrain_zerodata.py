from torch.utils.data import Dataset
import sys
from splitdata import split_train
sys.path.append('../')
from pretrain_dataset import prefmri_dataset


class pretrain_dataset(Dataset):
    def __init__(self,split = 'train',subj = 'subj01',test_num = 400,k = 2,seed = 42):
        super().__init__()
        dataset = prefmri_dataset('all', subj)
        dataset.set_k(k)
        self.train_set,self.test_set = split_train(dataset=dataset,test_num=test_num,seed=seed)
        # print(self.train_set.shape)
        self.k = dataset.k
        self.set_dataset(split)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
      return self.data[index]
    def set_dataset(self, split):
        if split == 'train':
            self.data = self.train_set
        elif split == 'test':
            self.data = self.test_set
        else:
            raise ValueError("Invalid split type! Use 'train' or 'test'.")
        # print(self.data)


