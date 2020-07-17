import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import config
from PIL import Image, ImageDraw2, ImageDraw



class xDataSet(Dataset):
    def __init__(self):
        self.OKroot = config.DataSetPath
        # self.NKroot = config.DataSetPath_
        self.images = [x.path for x in os.scandir(self.OKroot)]
        # self.images_ = [x.path for x in os.scandir(self.NKroot)]
        # print(self.images)
        # print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('L')
        img1 = np.array(img) / 127.5 - 1

        img2 = img
        box, color = self.getbox()
        ImageDraw.Draw(img2).ellipse(box, fill=color)
        img2 = np.array(img2)
        img2 = img2 / 127.5 - 1

        return torch.from_numpy(img2).unsqueeze(0), torch.from_numpy(img1).unsqueeze(0)


    @staticmethod
    def getbox():
        # x1 = np.random.randint(1, 127)
        # y1 = np.random.randint(1, 127)
        # x2 = x1 + np.random.randint(1, 75)
        # y2 = y1 + np.random.randint(1, 75)
        x1 = torch.randint(low=1, high=127, size=[1]).squeeze(0)
        y1 = torch.randint(low=1, high=127, size=[1]).squeeze(0)
        x2 = x1 + torch.randint(low=1, high=75, size=[1]).squeeze(0)
        y2 = y1 + torch.randint(low=1, high=75, size=[1]).squeeze(0)
        color = np.random.randint(5, 250)
        if x2 > 127:
            x2 = 127
        if y2 > 127:
            y2 = 127

        return (x1, y1, x2, y2), color

class tDataSet(Dataset):
    def __init__(self):
        self.NKroot = config.DataSetPath_
        self.images_ = [x.path for x in os.scandir(self.NKroot)]

    def __len__(self):
        return len(self.images_)

    def __getitem__(self, index):
        img = Image.open(self.images_[index]).convert('L')
        img1 = np.array(img) / 127.5 - 1

        return torch.from_numpy(img1).unsqueeze(0)