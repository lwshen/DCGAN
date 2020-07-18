import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import config
from PIL import Image, ImageDraw2, ImageDraw
import logging
from glob import glob


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

class unetDataset(Dataset):
    def __init__(self, imgs_dir, scale = 1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        x1 = np.random.randint(1, newW)
        y1 = np.random.randint(1, newH)
        x2 = x1 + np.random.randint(1, newW // 2)
        y2 = y1 + np.random.randint(1, newH // 2)
        color = np.random.randint(5, 250)
        if x2 > newW - 1:
            x2 = newW - 1
        if y2 > newH - 1:
            y2 = newH - 1
        box = (x1, y1, x2, y2)

        flaw = pil_img
        ImageDraw.Draw(flaw).ellipse(box, fill=color)
        img1 = np.array(flaw)

        mask = Image.new('L', (newW, newH), 0)
        ImageDraw.Draw(mask).ellipse(box, fill='white')
        img2 = np.array(mask)

        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=2)
        if len(img2.shape) == 2:
            img2 = np.expand_dims(img2, axis=2)

        # HWC to CHW
        img_trans1 = img1.transpose((2, 0, 1))
        if img_trans1.max() > 1:
            img_trans1 = img_trans1 / 255
        img_trans2 = img2.transpose((2, 0, 1))
        if img_trans2.max() > 1:
            img_trans2 = img_trans2 / 255

        return img_trans1, img_trans2

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0]).convert('L')

        img, mask = self.preprocess(img, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }