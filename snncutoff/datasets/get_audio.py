import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from torchvision import  transforms


class GetAudio(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        new_data = []
        # for t in range(data.size(0)):
        #     new_data.append(torch.tensor(data[t,...]))
        # data = torch.stack(new_data, dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data.transpose(0,1), target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))