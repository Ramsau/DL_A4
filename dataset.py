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
        positive = idx > len(self.files_neg)
        if positive:
            idx -= len(self.files_neg)
        dir = self.dir + ('/pos/' + self.files_pos[idx] if positive else '/neg/' + self.files_neg[idx])
        with open(dir, 'r') as file:
            text = file.read()
        return text, 1 if positive else 0



if __name__ == '__main__':
    dataset = IMDBDataset()
    test1 = dataset.__getitem__(0)
    test2 = dataset.__getitem__(13000)
    print(test1)
    print(test2)
