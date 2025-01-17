from torch.utils.data import DataLoader

from dataset import IMDBDataset


def main():
    dataset_train = IMDBDataset(train=True)
    dataset_test = IMDBDataset(train=False)
    dataloader = DataLoader(dataset_train)
    for batch in dataloader:
        print(batch)

if __name__ == '__main__':
    main()
