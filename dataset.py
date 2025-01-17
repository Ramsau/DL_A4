from torch.utils.data import Dataset
import os

class IMDBDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        if train:
            self.files_neg = os.listdir('./aclImdb/train/neg')
            self.files_pos = os.listdir('./aclImdb/train/pos')
            self.dir = './aclImdb/train'
        else:
            self.files_neg = os.listdir('./aclImdb/test/neg')
            self.files_pos = os.listdir('./aclImdb/test/pos')
            self.dir = './aclImdb/test'

    def __len__(self):
        return len(self.files_neg) + len(self.files_pos)

    def __getitem__(self, idx):
        if idx < len(self.files_neg):
            dir = self.dir + '/neg/' + self.files_neg[idx]
            label = 0
        else:
            idx -= len(self.files_neg)
            dir = self.dir + '/pos/' + self.files_pos[idx]
            label = 1

        with open(dir, 'r', encoding='utf-8') as file:
            text = file.read()
        return text, label
