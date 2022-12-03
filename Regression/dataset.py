import csv
import numpy as np
import random
import torch
from torch.utils.data import DataLoader


class HousingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 mode='train',
                 stats=None):

        if mode != 'train' and stats is None:
            raise ValueError('stats should not be none for dev or test mode!')
        if stats:
            mean, std, min, max = stats

            self.mean = mean
            self.std = std
            self.min = min
            self.max = max

        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:]).astype(float)

        # 测试
        if mode == 'test':
            data = data[:, [0, 1, 2]]
            data[:, 1] = np.log(data[:, 1])
            self.data = torch.FloatTensor(data)
        # 训练
        else:
            target = data[:, -1]
            data = data[:, [0, 1, 2]]
            data[:, 1] = np.log(data[:, 1])
            # 将训练数据进一步划分为训练集和验证集
            if mode == 'train':
                self.data = torch.FloatTensor(data)
                self.target = torch.log(torch.FloatTensor(target))

                self.mean = self.data.mean(dim=0, keepdim=True)
                self.std = self.data.std(dim=0, keepdim=True)
                self.min = self.data.min(dim=0, keepdim=True).values
                self.max = self.data.max(dim=0, keepdim=True).values

                # self.mean = self.data.min(dim=0, keepdim=True).values
                # self.std = self.data.max(dim=0, keepdim=True).values

            elif mode == 'dev':
                self.data = torch.FloatTensor(data)
                self.target = torch.log(torch.FloatTensor(target))
            else:
                raise Exception('Unknown mode!')

        # 数据归一化
        # self.data = (self.data - self.mean) / self.std
        # self.data = (self.data - self.mean) / (self.std - self.mean)
        self.data[:, [0, 1]] = (self.data[:, [0, 1]] - self.mean[:, [0, 1]]) / self.std[:, [0, 1]]
        self.data[:, 2] = (self.data[:, 2] - self.min[:, 2]) / (self.max[:, 2] - self.min[:, 2])

    def get_stats(self):
        return self.mean, self.std, self.min, self.max

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]  # 没有target

    def __len__(self):
        return len(self.data)


def build_dataloader(path, mode, stats=None, batch_size=4, num_workers=0, device='cpu'):
    dataset = HousingDataset(path=path, mode=mode, stats=stats)
    stats = dataset.get_stats()

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=(mode == 'train'),
                            drop_last=True,
                            num_workers=num_workers,
                            pin_memory=(device != 'cpu'))

    return dataloader, stats


if __name__ == '__main__':
    dataset, stats = build_dataloader('data/train.csv',
                                          batch_size=16,
                                          mode='train')

    print(stats)
