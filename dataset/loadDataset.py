from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from glob import glob
import torch


class LoadData(Dataset):
    """
    data loading demo
    """

    def __init__(self, data_root, std):
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        self.data_root = data_root
        self.list = self._get_image_list()
        self.std = std

    def _get_image_list(self):
        dirs = sorted(glob(self.data_root + '/*.png'))
        return dirs

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img = cv2.imread(self.list[index])
        gt = self.trsfm(img)
        noise = torch.FloatTensor(gt.size()).normal_(mean=0, std=self.std / 255.)
        noisy = gt + noise
        noisy = noisy.clamp(0, 1)
        return gt, noisy
